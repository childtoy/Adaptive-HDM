import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np
import random
import wandb
import pickle as pkl
import blobfile as bf
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.scripts.motion_process import recover_rot
import data_loaders.humanml.utils.paramUtil as paramUtil

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.skeleton import Skeleton, skel_joints
from data_loaders.tensors import collate
import pickle as pkl
from lpm.model import LengthPredctionUnet


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        # print(args.corr_noise)
        
        self.num_len = 0 
                        
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())        
        
        if args.corr_noise:
            with open(args.param_lenK_path, 'rb') as f: 
                self.param_lenK = pkl.load(f)
            self.num_len = len(self.param_lenK['K_param'])
            self.K_param = torch.Tensor(self.param_lenK['K_param']).to(self.device)
            template = self.param_lenK['template']
            self.template = torch.Tensor(template).repeat(self.batch_size, 1,1,1).to(self.device)
        else : 
            self.param_lenK = None

        self.schedule_sampler_type = 'uniform'        
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        # if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            # mm_num_samples = 0  # mm is super slow hence we won't run it during training
            # mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            # split=args.eval_split,
                                            # hml_mode='eval')

            # self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
            #                                        split=args.eval_split,
            #                                        hml_mode='gt')
            # self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            # self.eval_data = {
            #     'test': lambda: eval_humanml.get_mdm_loader(
            #         model, diffusion, args.eval_batch_size,
            #         gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
            #         args.eval_num_samples, scale=1.,
            #     )
            # }
        self.use_ddp = False
        self.ddp_model = self.model
        self.length_module = self._load_length_module()
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
                ,strict=False
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _load_length_module(self):
        
        model_pth = './save/final_lpm.pt'
        length_module = LengthPredctionUnet(
            name                 = 'unet',
            dims                 = 1,
            n_in_channels        = 1,
            n_base_channels      = 128,
            n_emb_dim            = 128,
            n_cond_dim           = 1,
            n_time_dim           = 0,
            n_enc_blocks         = 7, # number of encoder blocks
            n_groups             = 16, # group norm paramter
            n_heads              = 4, # number of heads in QKV attention
            actv                 = nn.SiLU(),
            kernel_size          = 3, # kernel size (3)
            padding              = 1, # padding size (1)
            use_attention        = False,
            skip_connection      = True, # additional skip connection
            chnnel_multiples     = [1,2,2,2,4,4,8],
            updown_rates         = [1,1,2,1,2,1,2],
            use_scale_shift_norm = True,
            device               = self.device,
        ) # input:[B x C x L] => output:[B x C x L]
        length_module.load_state_dict(torch.load(model_pth)['model_state_dict'])
        length_module.to(self.device)
        length_module.eval()
        self.cls_value = torch.Tensor(
            [0.033     , 0.14044444, 
             0.24788889, 0.35533333, 
             0.46277778, 0.67766667, 
             1.        ]).to(self.device)
        # self.cls_value = torch.Tensor([0.03, 0.12,   
        #                 0.21, 0.3,
        #                 0.39, 0.48,
        #                 0.57, 0.66,
        #                 0.8, 1.0])
        return length_module
    def _predict_length(self, motion, true_length=None):
        """
        :param motion: [B x n_joints x n_dims, 1, n_frames]
        :param true_length: [B x n_joints x n_dims,]
        """
        with torch.no_grad():
            if true_length is not None:
                true_length = true_length.float()
            output = self.length_module(motion, c=true_length)
        _, pred_idx = torch.max(output.data, 1)
        pred = self.cls_value[pred_idx]

        return pred, pred_idx

    def _cal_corr_mat(self, data_shape, pred_idx, K_param_bag, true_length, target_idx_array=[1,2]) : 
        # target idx_array : idx of joints which only used root xz
        pred_K_params = self.template.clone()
        if self.args.corr_mode == 'R_trsrot' : 
            pred_K_params[:, [0, 1, 2, 3, 193, 194, 195]] = K_param_bag[pred_idx].reshape(self.batch_size,-1,196,196)
        elif self.args.corr_mode == 'R_trs' : 
            pred_K_params[:,target_idx_array] = K_param_bag[pred_idx].reshape(self.batch_size,-1,196,196)
        elif self.args.corr_mode == 'all' :
            pred_K_params = K_param_bag[pred_idx].reshape(self.batch_size,-1,196,196)
        elif self.args.corr_mode == 'LP' :
            pred_K_params[:,4:67] = K_param_bag[pred_idx].reshape(self.batch_size,-1,196,196)
        elif self.args.corr_mode == 'LBody' :
            pred_K_params[:,4:22] = K_param_bag[pred_idx].reshape(self.batch_size,-1,196,196)
        return pred_K_params
        
    def run_loop(self):
        if self.args.wandb:
            wandb.init(project=self.args.project, name=self.args.save_dir)
            
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                
                motion = motion.to(self.device)       
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                if self.args.corr_noise :        
                    # B D 1 L 
                    B, D, _, L = motion.shape
                    if self.args.corr_mode == 'R_trsrot' : 
                        input_motion = motion[:,[0, 1, 2, 3, 193, 194, 195],:,:]
                        B_, D_, _, L_ = input_motion.shape
                        input_motion = input_motion.reshape(B_*D_, 1, L_)
                        true_length = cond['y']['lengths'].repeat_interleave(D_)
                        pred_lens, pred_idx = self._predict_length(input_motion, true_length)
                        pred_lens = pred_lens.reshape(B_, D_)
                        org_lens = (torch.ones([B,D])*0.033).to(self.device)
                        org_lens[:,[0, 1, 2, 3, 193, 194, 195]] = pred_lens
                    elif self.args.corr_mode == 'R_trs' : 
                        ######## root + rotation ######
                        joint_rotation = recover_rot(motion.squeeze().permute(0, 2, 1))   
                        input_motion = joint_rotation.view(joint_rotation.shape[0], joint_rotation.shape[1], -1).unsqueeze(-2).permute(0, 3, 2, 1)
                        B_, D_, _, L_ = input_motion.shape
                        input_motion = input_motion.reshape(B_*D_, 1, L_)
                        true_length = cond['y']['lengths'].repeat_interleave(D_)
                        pred_lens, pred_idx = self._predict_length(input_motion, true_length)
                        pred_lens = pred_lens.reshape(B_, D_)
                        org_lens = (torch.ones([B,D])*0.033).to(self.device)
                        org_lens[:,1:3] = pred_lens
                    elif self.args.corr_mode == 'all' : 
                        input_motion = motion[:,:,:,:]
                        B_, D_, _, L_ = input_motion.shape
                        input_motion = input_motion.reshape(B_*D_, 1, L_)
                        true_length = cond['y']['lengths'].repeat_interleave(D_)
                        pred_lens, pred_idx = self._predict_length(input_motion, true_length)
                        pred_lens = pred_lens.reshape(B_, D_)
                        # org_lens = np.ones([B,D])*0.033
                        org_lens = pred_lens
                    elif self.args.corr_mode == 'LP':
                        input_motion = motion[:,4:67,:,:]
                        B_, D_, _, L_ = input_motion.shape
                        input_motion = input_motion.reshape(B_*D_, 1, L_)
                        true_length = cond['y']['lengths'].repeat_interleave(D_)
                        pred_lens, pred_idx = self._predict_length(input_motion, true_length)
                        pred_lens = pred_lens.reshape(B_, D_)
                        org_lens = (torch.ones([B,D])*0.033).to(self.device)
                        org_lens[:,4:67] = pred_lens
                    elif self.args.corr_mode == 'LBody':
                        input_motion = motion[:,4:22,:,:]
                        B_, D_, _, L_ = input_motion.shape
                        input_motion = input_motion.reshape(B_*D_, 1, L_)
                        true_length = cond['y']['lengths'].repeat_interleave(D_)
                        pred_lens, pred_idx = self._predict_length(input_motion, true_length)
                        pred_lens = pred_lens.reshape(B_, D_)
                        org_lens = (torch.ones([B,D])*0.033).to(self.device)
                        org_lens[:,4:22] = pred_lens

                    # org_lens = pred_lens
                    self.pred_lens = org_lens
                    self.pred_idx = pred_idx.reshape(B_, D_)
                    # motion = motion.reshape(B, D, 1, L)
                    self.pred_K_param = self._cal_corr_mat(motion.shape, pred_idx, self.K_param, cond['y']['lengths'])
                else :
                    self.pred_lens = None
                    self.pred_idx = None
                    self.pred_K_param = None

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))
                            if self.args.wandb:
                                wandb.log({'Train loss':v}, step=self.step+self.resume_step)
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()
                    
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        
        collate_args = [{'inp': torch.zeros(196), 'tokens': None, 'lengths': 196}] * 1
        texts = ['a man moves forward']

        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        # collate_args = [{'inp': torch.zeros(60), 'tokens': None, 'lengths': 60}] * 1
        _, model_kwargs = collate(collate_args)
        all_motions = []
        all_lengths = []
        all_text = []

        if self.args.corr_noise : 
            eval_K_params = torch.zeros((2,263,196,196)).to(self.device) 
            eval_K_params[0,1:3] = self.K_param[0].repeat(2,1,1)
            eval_K_params[1,1:3] = self.K_param[-1].repeat(2,1,1)
            eval_len_param = torch.ones((2,263)).to(self.device) * 0.03
            eval_len_param[0,1:3] = torch.Tensor([0.033]).to(self.device).repeat(2)
            eval_len_param[1,1:3] = torch.Tensor([1.0]).to(self.device).repeat(2)
        else : 
            eval_K_params = None
            eval_len_param = None


        kinematic_chain = t2m_kinematic_chain
        
        ref_path = 'dataset/HumanML3D/new_joints/000000.npy'
        reference_data = torch.from_numpy(np.load(ref_path))
        reference_data = reference_data.reshape(len(reference_data), -1, 3)
        # (joints_num, 3)
        n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        kinematic_chain = t2m_kinematic_chain
        skeleton = Skeleton(n_raw_offsets, kinematic_chain, args.device)
    

        self.model.eval()
        for i in range(2):
            all_motions = []
            all_lengths = []
            all_text = []
            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(
                self.model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (1, self.model.njoints, self.model.nfeats, 196),  # BUG FIX
                eval_K_params[i].unsqueeze(0) if eval_K_params is not None else None,
                eval_len_param[i].unsqueeze(0) if eval_len_param is not None else None,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )        
            motion = sample.squeeze().permute(0,2,1)
            rot6d, r_pos = recover_from_rot(motion, 22)
            rot6d = rot6d.reshape(r_pos.shape[0],r_pos.shape[1],22,6)
            n_batch, n_frames, n_joints = 1, 196, 22
            
            rotation_quat = matrix_to_quaternion(
                cont6d_to_matrix(rot6d)
                )
            print(rotation_quat)
            rotation_quat = rotation_quat.reshape(64, n_frames, n_joints, 4)

            # root_pos = translation
            local_q_normalized = torch.nn.functional.normalize(rotation_quat, p=2.0, dim=-1)
            # global_pos, global_q = skeleton_smpl.forward_kinematics_with_rotation5(local_q_normalized, r_pos)
            joint_rot = local_q_normalized.cpu().numpy().reshape(n_batch*n_frames,n_joints,4)
            root_trj = r_pos.cpu().numpy().reshape(n_batch*n_frames,3)
                    
            new_shape = (n_batch*n_frames, n_joints, 3)
            skel_joint = torch.Tensor(skel_joints).to(args.device).expand(new_shape)
            # print(rotation_quat)
            new_joints = skeleton.forward_kinematics(rotation_quat.reshape(n_batch*n_frames,n_joints,4), r_pos.reshape(n_batch*n_frames,3), skel_joints=skel_joint)
            
            new_joints = new_joints.reshape(n_batch, n_frames, n_joints, 3)
            sample = new_joints
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            all_motions = np.concatenate(all_motions, axis=0)
            all_motions = all_motions[:1]  # [bs, njoints, 6, seqlen]
            all_text = all_text[:1]
            all_lengths = np.concatenate(all_lengths, axis=0)[:1]

            skeleton = paramUtil.t2m_kinematic_chain
            motion = all_motions[0].transpose(2, 0, 1)
            if i == 0 :
                plot_3d_motion(self.save_dir,'/eval_result_fast_'+str(self.step)+'.gif', skeleton, motion, dataset=self.args.dataset, title='length : 0.03', fps=20)
            else : 
                plot_3d_motion(self.save_dir,'/eval_result_slow_'+str(self.step)+'.gif', skeleton, motion, dataset=self.args.dataset, title='length : 1.0', fps=20)
        
        self.model.train()
        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
       
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                K_params=self.pred_K_param,
                len_param=self.pred_lens,
                model_kwargs=micro_cond,
                dataset=self.data.dataset,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)