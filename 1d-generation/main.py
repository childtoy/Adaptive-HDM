
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import math
import random 
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from util import (
    plot_ddpm_1d_result,
    get_hbm_M,
    box_plot
)
from diffusion import (
    get_ddpm_constants,
    DiffusionUNetLegacy,
    forward_sample,
    eval_ddpm_1d,
)
import argparse
from lpm.model import LengthPredctionUnet

np.set_printoptions(precision=3)
th.set_printoptions(precision=3)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)


def fix_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def main(args):
    fix_seed(2023)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    print ("PyTorch version:[%s]."%(th.__version__))

    dc = get_ddpm_constants(
        schedule_name = 'cosine', # 'linear', 'cosine'
        T             = 1000,
        np_type       = np.float32,
    )

    # Instantiate U-net
    device = 'cuda' # mps, cpu
    model = DiffusionUNetLegacy(
        name                 = 'unet',
        dims                 = 1,
        n_in_channels        = 1,
        n_base_channels      = 128,
        n_emb_dim            = 128,
        n_cond_dim           = 1,
        # n_time_dim           = 0,
        n_enc_blocks         = 5, # number of encoder blocks
        n_dec_blocks         = 5, # number of decoder blocks
        n_groups             = 16, # group norm paramter
        n_heads              = 4, # number of heads in QKV attention
        actv                 = nn.SiLU(),
        kernel_size          = 3, # kernel size (3)
        padding              = 1, # padding size (1)
        use_attention        = False,
        skip_connection      = True, # additional skip connection
        chnnel_multiples     = (1,2,2,4,8),
        updown_rates         = (1,1,1,1,1),
        use_scale_shift_norm = True,
        device               = device,
    ) # input:[B x C x L] => output:[B x C x L]

    lpm = LengthPredctionUnet(
                name                 = 'unet',
                length               = 60, 
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
                device               = device,
            ) # input:[B x C x L] => output:[B x C x L]

    lpm.load_state_dict(th.load('../save/final_lpm.pt')['model_state_dict'])
    lpm.eval()

    print ("Ready.")

    cls_value = th.Tensor([0.033     , 0.14044444, 
                    0.24788889, 0.35533333, 
                    0.46277778, 0.67766667, 
                    1.        ]).to(device)

    if args.test:
        model.load_state_dict(th.load(args.ckpt)['state_dict'])

    # Dataset
    train_data = np.load('./data/train_gp.npy', allow_pickle=True)
    train_data = train_data.item()
    train_x_0 = train_data['x_0']
    train_times = train_data['times']
    train_label = train_data['real_param']

    val_data = np.load('./data/val_gp.npy', allow_pickle=True)
    val_data = val_data.item()
    val_x_0 = val_data['x_0']
    val_times = val_data['times']
    val_label = val_data['real_param']

    train_normalized_data = (train_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())
    val_normalized_data = (val_x_0 - val_x_0.mean()) / math.sqrt(val_x_0.var())

    x_0 = train_normalized_data
    times = train_times
    label = train_label

    # M list
    M_list = []
    M_list.append(
        th.from_numpy(np.eye(len(times))).to(th.float32).to(device)
    )
    for hyp_len in cls_value:
        hyp_len = hyp_len.detach().cpu().numpy()
        M = get_hbm_M(times,hyp_gain=0.1,hyp_len=hyp_len,device=device) # [L x L]
        M_list.append(M) # length 11
    # M = None
    print ("Hilbert Brownian motion ready.")    
    output_pth = args.output_pth

    if args.train:
        # Configuration
        max_iter    = int(3e5)
        batch_size  = 128
        print_every = 1e2
        eval_every  = 1e2

        # Loop
        model.cuda().train()
        optm = th.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=0.0)
        schd = th.optim.lr_scheduler.ExponentialLR(optimizer=optm,gamma=0.99998)

        min_loss = 100_000
        ckpt_output = os.path.join(output_pth, 'ckpt')

        for it in range(max_iter):
            # Zero gradient
            optm.zero_grad()
            # Get batch
            idx = np.random.choice(x_0.shape[0],batch_size)
            x_0_batch = x_0[idx,:,:].to(device) # [B x C x L]
            cond_batch = label[idx][:, None].to(device)

            # prediction by LPM for condition of gp diffusion
            with th.no_grad():
                true_length = th.Tensor([196]*batch_size).to(device)
                output = lpm(x_0_batch, c=true_length)
                _, pred_idx = th.max(output.data, 1)
                cond_batch = cls_value[pred_idx][:, None]

            # Sample time steps
            step_batch = th.randint(0, dc['T'],(batch_size,),device=device).long() # [B]
            
            # Forward diffusion sampling
            M = None
            if args.corr:
                M = M_list
            x_t_batch,noise = forward_sample(x_0_batch,step_batch,dc,M) # [B x C x L]

            # Noise prediction
            noise_pred,_ = model(x_t_batch,step_batch,cond_batch) # [B x C x L]
            
            # Compute error
            loss = F.mse_loss(noise,noise_pred)+F.smooth_l1_loss(noise,noise_pred,beta=0.1)
            
            # Update
            loss.backward()
            optm.step()
            schd.step()
            
            # Print
            if (it%print_every) == 0 or it == (max_iter-1):
                print ("it:[%7d][%.1f]%% loss:[%.4f]"%(it,100*it/max_iter,loss.item()))
                if loss < min_loss:
                    if not os.path.exists(ckpt_output):
                        os.makedirs(ckpt_output)
                        
                    loss = round(loss.item(), 5)
                    th.save({
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optm.state_dict(),
                        'loss': loss
                    }, os.path.join(ckpt_output, f'it{it}_ckpt.pt'))
                min_loss = loss
                    
            # Evaluate
            if (it%eval_every) == 0 or it == (max_iter-1):
                iter_output = os.path.join(output_pth, str(it))
                
                if not os.path.exists(iter_output):
                    os.makedirs(iter_output)

                val_label = cls_value[val_label.long()]

                for hyp_len in [0.033, 0.24788889, 0.46277778, 1]:

                    print(f'Plotting {hyp_len}')
                    n_sample = 20

                    step_list_to_append = np.linspace(0,999,10).astype(np.int64) # save 10 x_ts
                    
                    M_eval = None
                    if args.corr:
                        M_eval = get_hbm_M(times,hyp_gain=0.1,hyp_len=hyp_len,device=device)
                    x_t_list = eval_ddpm_1d(
                        model = model,
                        dc = dc,
                        n_sample = n_sample,
                        x_0 = x_0,
                        step_list_to_append = step_list_to_append,
                        device = device,
                        M = M_eval,
                        hyp_len = th.Tensor([hyp_len]*n_sample).to(device).reshape(n_sample,1),
                        )

                    plot_ddpm_1d_result(
                        len_param=hyp_len, 
                        output=iter_output, 
                        real_param=val_label,
                        times=times,
                        x_data=val_normalized_data,
                        step_list=step_list_to_append,
                        x_t_list=x_t_list,
                        plot_ancestral_sampling=False,
                        plot_one_sample=False,
                        lw_gt=2,
                        lw_sample=1/2,
                        lc_gt=(0,1,0,0.3),
                        lc_sample='k',
                        ls_gt='-',
                        ls_sample='-',
                        ylim=(-4,+4),
                        figsize=(6,4)
                        )

        print ("Done.\n")

        th.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optm.state_dict(),
            'loss': loss
        }, os.path.join(ckpt_output, 'last_ckpt.pt'))

    if args.test:
        ckpt_name = args.ckpt.split('/')[-1].split('_')[0]
        name = f'test_{ckpt_name}'
        iter_output = os.path.join(output_pth, name)
        cls_value = th.Tensor(
                [0.033     , 0.14044444, 
                0.24788889, 0.35533333, 
                0.46277778, 0.67766667, 
                1.        ]
            ).to(device)        
        x_0 = val_x_0
        times = val_times
        label = cls_value[val_label.long()]

        if not os.path.exists(iter_output):
            os.makedirs(iter_output)
        
        x_list = []
        total_velocity = {} 
        total_mean_velocity = {}

        for hyp_len in [0.033     , 0.14044444, 
                        0.24788889, 0.35533333, 
                        0.46277778, 0.67766667, 
                        1.        ]:
            print(f'Plotting {hyp_len}')
            # hyp_len = hyp_len.detach().cpu().numpy()
            n_sample = 100

            step_list_to_append = np.linspace(0,999,10).astype(np.int64) # save 10 x_ts
            
            M_eval=None
            
            if args.corr:
                M_eval = get_hbm_M(times,hyp_gain=0.1,hyp_len=hyp_len,device=device)

            x_t_list = eval_ddpm_1d(
                        model = model,
                        dc = dc,
                        n_sample = n_sample,
                        x_0 = x_0,
                        step_list_to_append = step_list_to_append,
                        device = device,
                        M = M_eval,
                        hyp_len = th.Tensor([hyp_len]*n_sample).to(device).reshape(n_sample,1),
                        )
            
            pred = x_t_list[0].squeeze()
            velocity = th.abs(pred[:, :-1] - pred[:, 1:])
            total_velocity[hyp_len] = (th.sum(velocity, dim=-1) / len(velocity)).detach().cpu().numpy()
            total_mean_velocity[hyp_len] = (th.mean(th.sum(velocity, dim=-1)/len(velocity))).detach().cpu().numpy()

            plot_ddpm_1d_result(
                len_param=hyp_len, 
                output=iter_output, 
                real_param=label,
                times=times,
                x_data=x_0,
                step_list=step_list_to_append,
                x_t_list=x_t_list,
                plot_ancestral_sampling=False,
                plot_one_sample=False,
                lw_gt=2,
                lw_sample=1/2,
                lc_gt=(0,1,0,0.3),
                lc_sample='k',
                ls_gt='-',
                ls_sample='-',
                ylim=(-1.5,+1.5),
                figsize=(6,4)
                )
        
        params = [0.033     , 0.14044444, 
                0.24788889, 0.35533333, 
                0.46277778, 0.67766667, 
                1.        ]
        
        print(f" ========================== ckpt {ckpt_name} ========================")
        for i in range(len(params)):
            mu = total_velocity[params[i]].mean()
            sigma = total_velocity[params[i]].std()
            
            print(f' * {round(params[i],3)}: \t mu {mu}, sigma {sigma}')
        print(" ====================================================================")

        item = list(total_velocity.values())
        box_plot(item, f'{iter_output}_box')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDM 1D")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model", type=str, default='unet')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pb_start", type=float, default=1e-3)
    parser.add_argument("--pb_end", type=float, default=1e-2)
    parser.add_argument("--inb", action='store_true')
    parser.add_argument("--corr", action='store_true')
    parser.add_argument("--output_pth", type=str, default='./')
    parser.add_argument("--ckpt", type=str, default='./')
    args = parser.parse_args()
    
    main(args)