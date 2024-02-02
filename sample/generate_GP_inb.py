# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
import pickle as pkl

from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, sk_parents, sk_skeleton_part)
from data_loaders.humanml.common.quaternion import cont6d_to_quat
import sys 

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 60
    fps = 30 
    n_frames = 60
    
    # with open(args.param_lenK_path, 'rb') as f : 
        # param_lenK = pkl.load(f)
        
    param_lenK = None
    
    dist_util.setup_dist(args.device)
    
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_nocorr_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    # collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                        # arg, one_action, one_action_text in zip(collate_args, action, action_text)]
    _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []
    if param_lenK is not None : 
        K_params = param_lenK['K_param'][args.len_idx]
        print('# of len params :', len(K_params))
        len_param = param_lenK['len_param'][args.len_idx]
        len_param = torch.Tensor([len_param]).to(dist_util.dev()).repeat(args.batch_size,1).reshape(args.batch_size,1)
    else : 
        K_params = None
        len_param = None

    offset = sk_offsets 
    skeleton_mocap = Skeleton(offsets=offset, parents=sk_parents, device=args.device)
    skeleton_mocap.remove_joints(sk_joints_to_remove)

    skeleton_mocap._parents


    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        sample_fn = diffusion.p_sample_loop
        print(diffusion.model_mean_type)
        sample = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            K_params,
            len_param,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        
        sample = sample[rep_i].permute(2,1,0).reshape(60,23,6).unsqueeze(0)
        offset = sk_offsets 
        skeleton_mocap = Skeleton(offsets=offset, parents=sk_parents, device=args.device)
        skeleton_mocap.remove_joints(sk_joints_to_remove)
        skeleton_mocap._parents
        # sample = torch.Tensor(input_data).unsqueeze(0)
        # sample = input_data
        root_pos = sample[:,:,-1,:3]
        rot_6d = sample[:,:,:-1,:]
        # print(rot_6d.shape)
        rot_quat = cont6d_to_quat(torch.Tensor(rot_6d).to(args.device))
        # # rot_quat = rot_quat.reshape(-1,60,22,4)
        local_q_normalized = torch.nn.functional.normalize(rot_quat, p=2.0, dim=-1)
        global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, torch.Tensor(root_pos).to(args.device))
        global_pos_tmp = global_pos
        global_pos[:,:,:,2] = global_pos_tmp[:,:,:,1]
        global_pos[:,:,:,1] = global_pos_tmp[:,:,:,2]
        caption = 'unconditioned_LAFAN'
        length = 60
        # motion = global_pos[0].cpu().numpy().transpose(2, 0, 1)
        motion = global_pos[0].cpu().numpy()
        animation_save_path = os.path.join(out_path, 'result_inb-'+str(rep_i)+'.gif')
        plot_3d_motion(animation_save_path, sk_skeleton_part, motion, dataset='LAFAN', title=caption, fps=30)
        
    # all_motions = np.concatenate(all_motions, axis=0)
    # all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    # all_text = all_text[:total_num_samples]
    # all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    # npy_path = os.path.join(out_path, 'results.npy')
    # print(f"saving results file to [{npy_path}]")
    # np.save(npy_path,
    #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
    #          'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    # with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
    #     fw.write('\n'.join(all_text))
    # with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
    #     fw.write('\n'.join([str(l) for l in all_lengths]))

    # print(f"saving visualizations to [{out_path}]...")
    # skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # sample_files = []
    # num_samples_in_out_file = 7



    # for sample_i in range(args.num_samples):
    #     rep_files = []
    #     for rep_i in range(args.num_repetitions):
    #         caption = all_text[rep_i*args.batch_size + sample_i]
    #         length = all_lengths[rep_i*args.batch_size + sample_i]
    #         motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
    #         save_file = sample_file_template.format(sample_i, rep_i)
    #         print(sample_print_template.format(caption, sample_i, rep_i, save_file))
    #         animation_save_path = os.path.join(out_path, save_file)
    #         plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
    #         # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
    #     #     rep_files.append(animation_save_path)

    #     # sample_files = save_multiple_samples(args, out_path,
    #     #                                        row_print_template, all_print_template, row_file_template, all_file_template,
    #     #                                        caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    # abs_path = os.path.abspath(out_path)
    # print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.gif'
    all_file_template = 'samples_{:02d}_to_{:02d}.gif'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.gif'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.gif'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
