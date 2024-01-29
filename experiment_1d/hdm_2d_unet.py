import os
import wandb
import random
import argparse
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from util import (
    print_model_parameters,
    gp_sampler,
    get_torch_size_string,
    plot_1xN_torch_traj_tensor,
    plot_ddpm_1d_result,
    plot_ddpm_2d_data_result,
    get_hbm_M,
    get_hbm_M_2d
)
from dataset import get_1d_training_data, get_2d_training_data
from diffusion import (
    get_ddpm_constants,
    plot_ddpm_constants,
    DiffusionUNetLegacy,
    DiffusionUNetLegacy_leng,
    DiffusionUNetLegacy_origin,
    forward_sample,
    eval_ddpm_1d,
)
from dataset import mnist

def main(args):
    time = dt.datetime.now()
    print(f'> > Train START {time.hour}:{time.minute}:{time.second} \n')

    np.set_printoptions(precision=3)
    th.set_printoptions(precision=3)
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)

    # %matplotlib inline
    # %config InlineBackend.figure_format='retina'
    print ("PyTorch version:[%s]."%(th.__version__))

    dc = get_ddpm_constants(
        schedule_name = 'cosine', # 'linear', 'cosine'
        T             = 1000,
        np_type       = np.float32,
    )
    # for k_idx,key in enumerate(dc.keys()): print ("[%2d] key:[%s]"%(k_idx,key))
    # plot_ddpm_constants(dc)

    output = f'./result/2d_example_result_{args.name}'
    if not os.path.exists(output):
        os.makedirs(output)

    # Length param
    slow_len_param = [0.001, 0.01, 0.06, 0.1, 0.2]
    len_param = list(map(float, args.len_param.split(',')))
    channel = list(map(int, args.channel.split(',')))
    rate = list(map(int, args.rate.split(',')))
    
    # Instantiate U-net
    if th.cuda.is_available():
        device =  'cuda' # mps, cpu
    
    if args.true:
        model = DiffusionUNetLegacy_leng(
            name                 = 'unet',
            dims                 = 1,
            n_in_channels        = 2,
            n_base_channels      = 128,
            n_emb_dim            = args.feature,
            n_cond_dim           = 2,
            n_leng_cond_dim      = 2,
            n_enc_blocks         = args.block, # number of encoder blocks
            n_dec_blocks         = args.block, # number of decoder blocks
            n_groups             = 16, # group norm paramter
            n_heads              = 4, # number of heads in QKV attention
            actv                 = nn.SiLU(),
            kernel_size          = 3, # kernel size (3)
            padding              = 1, # padding size (1)
            use_attention        = False,
            skip_connection      = True, # additional skip connection
            chnnel_multiples     = channel,
            updown_rates         = rate,
            use_scale_shift_norm = True,
            device               = device,
        ) # input:[B x C x L] => output:[B x C x L]
        
    else:
        model = DiffusionUNetLegacy(
            name                 = 'unet',
            dims                 = 1,
            n_in_channels        = 2,
            n_base_channels      = 128,
            n_emb_dim            = args.feature,
            n_cond_dim           = 1,
            n_enc_blocks         = args.block, # number of encoder blocks
            n_dec_blocks         = args.block, # number of decoder blocks
            n_groups             = 16, # group norm paramter
            n_heads              = 4, # number of heads in QKV attention
            actv                 = nn.SiLU(),
            kernel_size          = 3, # kernel size (3)
            padding              = 1, # padding size (1)
            use_attention        = False,
            skip_connection      = True, # additional skip connection
            chnnel_multiples     = channel,
            updown_rates         = rate,
            use_scale_shift_norm = True,
            device               = device,
        ) # input:[B x C x L] => output:[B x C x L]
        
    print ("Ready.")

    if args.test == True and args.train == False:
        model.load_state_dict(th.load(os.path.join(output, 'ckpt.pt'))['model_state_dict'])
        
    # times, x_0, _, true_length, real_param = get_2d_training_data(
    #     traj_type = args.d_type, # gp / step / step2
    #     n_traj    = args.size,
    #     L         = 128,
    #     device    = device,
    #     seed      = 0,
    #     plot_data = True,
    #     figsize   = (6,3),
    #     ls        = '-',
    #     lc        = 'k',
    #     lw        = 1/2,
    #     verbose   = True,
    #     output    = output,
    #     len_param = len_param
    #     )
    
    # import ipdb;ipdb.set_trace()
    # dic = {'times': times, 'x_0': x_0, 'true_length':true_length, 'real_param':real_param}
    # np.save('2d_gp3_ds1000_small_slow.npy', dic)
    
    if args.d_type == 'gp1': # 두 dim 모두 동일
        data = np.load('./2d_gp1_ds1000.npy', allow_pickle=True)
        data = data.item()
        times = data['times']
        x_0 = data['x_0']
        true_length = data['true_length']
        real_param = data['real_param']

    elif args.d_type == 'gp2': # 같은 length param에서 sampling
        data = np.load('./2d_gp2_ds1000.npy', allow_pickle=True)
        data = data.item()
        times = data['times']
        x_0 = data['x_0']
        true_length = data['true_length']
        real_param = data['real_param']
    
    elif args.d_type == 'gp3': # 서로 다른 length param에서 sampling
        data = np.load('./2d_gp3_ds1000_small_slow.npy', allow_pickle=True)
        data = data.item()
        times = data['times']
        x_0 = data['x_0']
        true_length = data['true_length']
        real_param = data['real_param']
    
    # temp = x_0[:, 0, :].unsqueeze(1)
    # x_0 = th.cat([x_0, temp], dim=1)[:, 1:, :]
    
    # temp = real_param[:, 0].unsqueeze(1)
    # real_param = th.cat([real_param, temp], dim=1)[:, 1:]
    
    if args.train:
        # Multi-resolution Hilbert Spaces        
        M_list = []
        M_list.append(
            th.from_numpy(np.eye(len(times))).to(th.float32).to(device)
        )
        for hyp_len in len_param:
            M = get_hbm_M(times,hyp_gain=1.0,hyp_len=hyp_len,device=device) # [L x L]
            M_list.append(M)
                        
        # M = None
        print ("Hilbert Brownian motion ready.\n")

        # Configuration
        max_iter    = int(2e4)
        batch_size  = 128
        print_every = 1e3
        eval_every  = 1e3

        # Train Loop
        model.train()
        optm = th.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=0.0)
        schd = th.optim.lr_scheduler.ExponentialLR(optimizer=optm,gamma=0.99998)

        for it in range(max_iter):
            
            # Zero gradient
            optm.zero_grad()
            
            # Get batch
            idx = np.random.choice(x_0.shape[0],batch_size)
            x_0_batch = x_0[idx,:,:] # [B x C x L]
            # cond_batch = real_param[idx, :] # [B x C]
            
            if args.d_type == 'gp1':
                cond_idx = np.random.randint(0,7)
                cond_batch = th.Tensor(len_param).to(device).repeat(128,1)[:,cond_idx].reshape(128,1)
                cond_batch = th.cat([cond_batch]*2, dim=1)
            
            elif args.d_type == 'gp2':
                # 손목
                cond_idx = np.random.randint(0,7)
                cond_batch_up = th.Tensor(len_param).to(device).repeat(128,1)[:,cond_idx].reshape(128,1)
                
                # 발목
                cond_idx = np.random.randint(0,7)
                cond_batch_down = th.Tensor(len_param).to(device).repeat(128,1)[:,cond_idx].reshape(128,1)
                cond_batch = th.cat([cond_batch_up, cond_batch_down], dim=1)
            
            elif args.d_type == 'gp3':
                # 손목
                cond_idx = np.random.randint(0,7)
                cond_batch_up = th.Tensor(len_param).to(device).repeat(128,1)[:,cond_idx].reshape(128,1)
                
                # 발목
                cond_idx = np.random.randint(0,5)
                cond_batch_down = th.Tensor(slow_len_param).to(device).repeat(128,1)[:,cond_idx].reshape(128,1)
                cond_batch = th.cat([cond_batch_up, cond_batch_down], dim=1)
                
            # Sample time steps
            step_batch = th.randint(0, dc['T'],(batch_size,),device=device).long() # [B]
            
            # Forward diffusion sampling
            x_t_batch,noise = forward_sample(x_0_batch, step_batch, dc, M_list) # [B x C x L]
            
            # Noise prediction
            
            if args.true:
                true_length_batch = true_length[idx]
                noise_pred,_ = model(x_t_batch,step_batch,cond_batch, true_length_batch)    
            
            else:
                noise_pred,_ = model(x_t_batch,step_batch,cond_batch) # [B x C x L]
            
            # Compute error
            loss = F.mse_loss(noise,noise_pred)+F.smooth_l1_loss(noise,noise_pred,beta=0.1)
            wandb.log({'Train loss (MSE)': loss})
            
            # Update
            loss.backward()
            optm.step()
            schd.step()
            
            # Print
            if (it%print_every) == 0 or it == (max_iter-1):
                print ("it:[%7d][%.1f]%% loss:[%.4f]"%(it,100*it/max_iter,loss.item()))
            
            # Evaluate
            if (it%eval_every) == 0 or it == (max_iter-1):
                iter_output = os.path.join(output, str(it))
                if not os.path.exists(iter_output):
                    os.makedirs(iter_output)
                
                if args.d_type == 'gp1' or args.d_type == 'gp2':
                    for hyp_len in len_param:
                        n_sample = 20
                        leng = 0
                        if args.true:
                            leng = th.Tensor([128]*n_sample).to(device)

                        step_list_to_append = np.linspace(0,999,10).astype(np.int64) # save 10 x_ts
                        M_eval = get_hbm_M(times,hyp_gain=1.0,hyp_len=hyp_len,device=device)
                        
                        hyp_len_set = th.cat([th.Tensor([hyp_len]*n_sample).to(device).reshape(n_sample,1)]*2, dim=1)
                        
                        x_t_list = eval_ddpm_1d(
                            model = model,
                            dc = dc,
                            true = args.true,
                            n_sample = n_sample,
                            true_length = leng,
                            x_0 = x_0,
                            step_list_to_append = step_list_to_append,
                            device = device,
                            cond = None,
                            M = M_eval,
                            hyp_len = hyp_len_set)
                        
                        plot_ddpm_2d_data_result(
                            len_param=(hyp_len, hyp_len), output=iter_output, real_param=real_param,
                            times=times,x_data=x_0,step_list=step_list_to_append,x_t_list=x_t_list,
                            plot_ancestral_sampling=False,plot_one_sample=False,
                            lw_gt=2,lw_sample=1/2,lc_gt=(0,1,0,0.3),lc_sample='k',
                            ls_gt='-',ls_sample='-',ylim=(-4,+4),figsize=(15,5))
                
                elif args.d_type == 'gp3':
                    for hyp_len in len_param:
                        for slow_hyp_len in slow_len_param:
                            print (f"{it} steps | hyp_len: {hyp_len:.3e} {slow_hyp_len:.3e}")
                            n_sample = 20
                            leng = 0
                            if args.true:
                                leng = th.Tensor([128]*n_sample).to(device)

                            step_list_to_append = np.linspace(0,999,10).astype(np.int64) # save 10 x_ts
                            M_eval = get_hbm_M(times,hyp_gain=1.0,hyp_len=hyp_len,device=device)
                            
                            hyp_len_set = th.Tensor([hyp_len]*n_sample).to(device).reshape(n_sample,1)
                            slow_hyp_len_set = th.Tensor([slow_hyp_len]*n_sample).to(device).reshape(n_sample,1)
                            hyp_len_set = th.cat([hyp_len_set, slow_hyp_len_set], dim=1)
                                
                            x_t_list = eval_ddpm_1d(
                                model = model,
                                dc = dc,
                                true = args.true,
                                n_sample = n_sample,
                                true_length = leng,
                                x_0 = x_0,
                                step_list_to_append = step_list_to_append,
                                device = device,
                                cond = True,
                                M = M_eval,
                                hyp_len = hyp_len_set)
                            
                            plot_ddpm_2d_data_result(
                                len_param=(hyp_len, slow_hyp_len), output=iter_output, real_param=real_param,
                                times=times,x_data=x_0,step_list=step_list_to_append,x_t_list=x_t_list,
                                plot_ancestral_sampling=False,plot_one_sample=False,
                                lw_gt=2,lw_sample=1/2,lc_gt=(0,1,0,0.3),lc_sample='k',
                                ls_gt='-',ls_sample='-',ylim=(-4,+4),figsize=(25,5))
                        

        print ("Done.\n")

        th.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optm.state_dict(),
            'loss': loss
        }, os.path.join(output, 'ckpt.pt'))
        
    if args.test:     
        if args.d_type == 'gp1' or args.d_type == 'gp2':
            for hyp_len in len_param:
                n_sample = 20
                leng = 0
                if args.true:
                    leng = th.Tensor([128]*n_sample).to(device)

                step_list_to_append = np.linspace(0,999,10).astype(np.int64) # save 10 x_ts
                M_eval = get_hbm_M(times,hyp_gain=1.0,hyp_len=hyp_len,device=device)
                
                hyp_len_set = th.cat([th.Tensor([hyp_len]*n_sample).to(device).reshape(n_sample,1)]*2, dim=1)
                
                x_t_list = eval_ddpm_1d(
                    model = model,
                    dc = dc,
                    true = args.true,
                    n_sample = n_sample,
                    true_length = leng,
                    x_0 = x_0,
                    step_list_to_append = step_list_to_append,
                    device = device,
                    cond = None,
                    M = M_eval,
                    hyp_len = hyp_len_set)
                
                plot_ddpm_2d_data_result(
                    len_param=(hyp_len, hyp_len), output=iter_output, real_param=real_param,
                    times=times,x_data=x_0,step_list=step_list_to_append,x_t_list=x_t_list,
                    plot_ancestral_sampling=False,plot_one_sample=False,
                    lw_gt=2,lw_sample=1/2,lc_gt=(0,1,0,0.3),lc_sample='k',
                    ls_gt='-',ls_sample='-',ylim=(-4,+4),figsize=(15,5))
        
        elif args.d_type == 'gp3':
            for hyp_len in len_param:
                for slow_hyp_len in slow_len_param:
                    print (f"hyp_len: {hyp_len:.3e} {slow_hyp_len:.3e}")
                    n_sample = 20
                    leng = 0
                    if args.true:
                        leng = th.Tensor([128]*n_sample).to(device)

                    step_list_to_append = np.linspace(0,999,10).astype(np.int64) # save 10 x_ts
                    M_eval = get_hbm_M(times,hyp_gain=1.0,hyp_len=hyp_len,device=device)
                    
                    hyp_len_set = th.Tensor([hyp_len]*n_sample).to(device).reshape(n_sample,1)
                    slow_hyp_len_set = th.Tensor([slow_hyp_len]*n_sample).to(device).reshape(n_sample,1)
                    hyp_len_set = th.cat([slow_hyp_len_set, hyp_len_set], dim=1)
                        
                    x_t_list = eval_ddpm_1d(
                        model = model,
                        dc = dc,
                        true = args.true,
                        n_sample = n_sample,
                        true_length = leng,
                        x_0 = x_0,
                        step_list_to_append = step_list_to_append,
                        device = device,
                        cond = True,
                        M = M_eval,
                        hyp_len = hyp_len_set)
                    
                    plot_ddpm_2d_data_result(
                        len_param=(slow_hyp_len, hyp_len), output=output, real_param=real_param,
                        times=times,x_data=x_0,step_list=step_list_to_append,x_t_list=x_t_list,
                        plot_ancestral_sampling=False,plot_one_sample=False,
                        lw_gt=2,lw_sample=1/2,lc_gt=(0,1,0,0.3),lc_sample='k',
                        ls_gt='-',ls_sample='-',ylim=(-4,+4),figsize=(25,5))
                
        time = dt.datetime.now()
        print(f'\n> > Train END {time.hour}:{time.minute}:{time.second}')

if __name__ == "__main__":
    def fix_seed(seed):
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    fix_seed(2023)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="HDM 1D")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--true", action='store_true')
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--name", type=str, default='ds100_len01')
    parser.add_argument("--d_type", type=str, default='gp2')
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--len_param", type=str, default='0.001, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0')
    parser.add_argument("--channel", type=str, default='1,2,2,4,8')
    parser.add_argument("--rate", type=str, default='1,2,2,2,2')
    args = parser.parse_args()
    
    wandb.init(project='hdm-1d', name=args.name)

    main(args)