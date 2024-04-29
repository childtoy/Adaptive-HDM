# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from workspace.childtoy.MotionGPDiffusion.utils.model_util_base import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
import pickle as pkl
from Motion.transforms import repr6d2quat
from Motion import BVH
from Motion.Animation import positions_global as anim_pos
from Motion.Animation import Animation
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Quaternions import Quaternions
from sklearn.preprocessing import LabelEncoder
from data_loaders.tensors import collate

from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, sk_parents, sk_skeleton_part)
from data_loaders.humanml.common.quaternion import cont6d_to_quat
import sys 

def main():
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 60
    fps = 30
    n_frames = 60
    
    bvh_anim, joint_names, frametime = BVH.load(args.reference_bvh_path)
    data_path = '/workspace/workspace/childtoy/MotionGPDiffusion/dataset/LAFAN/full_data_pred_root_norm0.pkl'
    with open(data_path, 'rb') as f : 
        trained_data = pkl.load(f)
    lens_array = trained_data['lens_array']
    init_motion = trained_data['input_rnorm'][0]
    # print(';la;sldkj;lakjf', init_motion.shape)
    init_motion[1:-1] = 0 
    with open('./LAFAN_K_param_data60_fps30_dim138_len10.pkl', 'rb') as f : 
        param_lenK = pkl.load(f)
    
    num_len = len(param_lenK['K_param'])

    dist_util.setup_dist(args.device)
    
    
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_upper_foot_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    # model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    # collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                        # arg, one_action, one_action_text in zip(collate_args, action, action_text)]
    _, model_kwargs = collate(collate_args)

    # iterator = iter(data)
    # input_motions, model_kwargs = next(iterator)
    init_motion = trained_data['input_rnorm'][0]
    init_motion = torch.Tensor(init_motion).to(args.device).unsqueeze(0).permute(0,2,3,1).reshape(args.batch_size,model.njoints, model.nfeats, max_frames)
    input_motions = init_motion.to(dist_util.dev())
    texts = [''] * args.num_samples
    model_kwargs['y']['text'] = texts
    # if args.text_condition == '':
    args.guidance_param = 0.  # Force unconditioned generation

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    # if args.edit_mode == 'in_between':
    model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                            device=input_motions.device)  # True means use gt motion
    for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
        start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
        gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
        model_kwargs['y']['inpainting_mask'][i, :, :,start_idx: end_idx] = False  # do inpainting in those frames
    # elif args.edit_mode == 'upper_body':
    # model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
    #                                                     device=input_motions.device)  # True is lower body data
    # model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
    #     -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

    all_motions = []
    all_lengths = []
    all_text = []

    round_lens = lens_array.reshape(-1).round(2)
    le = LabelEncoder()
    le.fit(round_lens)
    lens_array = le.transform(round_lens)
    lens_array = lens_array.reshape(-1,23,6)
    # min_lens_array = lens_array[10096]
    max_lens_array = lens_array[31134] 
    target_lens_array = max_lens_array
    print(target_lens_array.shape)
    # middle_lens_array = lens_array[3]   
    specific_idx = 10
    # target_lens_array = lens_array[specific_idx]
    # print(min_lens_array.shape)
    length_can = le.classes_
    gen_list = []
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    for can_idx, last_len in enumerate(length_can) : 
        K_param = torch.zeros([args.batch_size, 23, 6, 60, 60]).to(args.device)
        len_param = torch.zeros([args.batch_size, 23, 6]).to(args.device)
        for i in range(args.batch_size):
            for j in range(23):
                for k in range(6):
                    # if j == 22 :
                    if j == 3 or j == 7  :
                        K_param[i, j, k] = torch.Tensor(param_lenK['K_param'][can_idx]).to(args.device)
                        len_param[i,j,k] = param_lenK['len_param'][can_idx]
                        # if k >3 : 
                        #     K_param[i, j, k] = torch.Tensor(param_lenK['K_param'][0]).to(args.device)
                        #     len_param[i,j,k] = param_lenK['len_param'][0]
                    else :       
                        index = int(target_lens_array[j, k])
                        K_param[i, j, k] = torch.Tensor(param_lenK['K_param'][index]).to(args.device)
                        len_param[i,j,k] = param_lenK['len_param'][index]
        K_param = K_param.reshape(args.batch_size, -1, 60, 60)
        len_param = len_param.reshape(args.batch_size, -1)        
        offset = sk_offsets 
        skeleton_mocap = Skeleton(offsets=offset, parents=sk_parents, device=args.device)
        skeleton_mocap.remove_joints(sk_joints_to_remove)

        skeleton_mocap._parents
        args.num_repetitions = 1
        for rep_i in range(args.num_repetitions):
            print(f'### Start sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            samples = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                K_param,
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
            for batch_i in range(args.batch_size):
                sample = samples[batch_i].permute(2,1,0).reshape(60,23,6).unsqueeze(0)
                offset = sk_offsets 
                skeleton_mocap = Skeleton(offsets=offset, parents=sk_parents, device=args.device)
                skeleton_mocap.remove_joints(sk_joints_to_remove)
                skeleton_mocap._parents
                # sample = torch.Tensor(input_data).unsqueeze(0)
                # sample = input_data
                mean_root = torch.Tensor(trained_data['mean_root']).to(args.device)
                std_root = torch.Tensor(trained_data['std_root']).to(args.device)
                root_pos = sample[:,:,-1,:3] * std_root +  mean_root.unsqueeze(0)
                # root_pos = sample[:,:,-1,:3]
                rot_6d = sample[:,:,:-1,:]
                # print(rot_6d.shape)
                rot_quat = cont6d_to_quat(rot_6d)
                # # rot_quat = rot_quat.reshape(-1,60,22,4)
                local_q_normalized = torch.nn.functional.normalize(rot_quat, p=2.0, dim=-1)
                global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, root_pos)
                global_pos_tmp = global_pos
                global_pos[:,:,:,2] = global_pos_tmp[:,:,:,1]
                global_pos[:,:,:,1] = global_pos_tmp[:,:,:,2]
                caption = 'slow_length_array adjust foot len : ' + str(last_len)
                length = 60
                # motion = global_pos[0].cpu().numpy().transpose(2, 0, 1)
                motion = global_pos[0].cpu().numpy()
                print('last_len')
                
                
                # bvh_path = os.path.join(out_path, f'sample{i:02d}.bvh')
                # one_sample = one_sample.reshape(n_frames, -1, joint_features_length)
                # quats = repr6d2quat(torch.tensor(one_sample[:, :, 3:])).numpy()
                # print(root_pos.shape)
                # print(local_q_normalized.shape)
                # anim = Animation(rotations=Quaternions(local_q_normalized[0].cpu().numpy()), positions=motion,
                                    # orients=bvh_anim.orients, offsets=bvh_anim.offsets, parents=bvh_anim.parents)
                # bvh_path = os.path.join(out_path, 'result_slowlen_inb-'+str(last_len)+'-'+str(batch_i)+'.bvh')
                # BVH.save(os.path.expanduser(bvh_path), anim, joint_names, frametime, positions=True)  # "positions=True" is important for the dragon and does not harm the others
                
                animation_save_path = os.path.join(out_path, 'result_slowlen_foot_inb-'+str(last_len)+'-'+str(batch_i)+'.gif')
                plot_3d_motion(animation_save_path, sk_skeleton_part, motion, dataset='LAFAN', title=caption, fps=20)
                gen_list.append(motion)
                    
    with open(os.path.join(out_path, 'max_rootjoint_change.pkl'), 'wb') as f :
        pkl.dump(gen_list, f)
        


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    # if args.dataset in ['kit', 'humanml']:
        # data.dataset.t2m_dataset.fixed_length = n_frames
    return data

if __name__ == "__main__":
    main()
