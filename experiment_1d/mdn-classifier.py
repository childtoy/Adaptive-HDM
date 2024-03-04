import os
import sys
sys.path.append('/nas/home/drizzle0171/motion-generation/MotionGPDiffusion')

import math
import wandb
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import (
    print_model_parameters,
    gp_sampler,
    get_torch_size_string,
    plot_1xN_torch_traj_tensor,
    plot_ddpm_1d_result,
    get_hbm_M,
)
from dataset import get_1d_training_data
from diffusion import (
    get_ddpm_constants,
    plot_ddpm_constants,
    DiffusionUNetLegacy,
    forward_sample,
    eval_ddpm_1d,
)

from LPM.model import (
    get_ddpm_constants,
    forward_sample,
)

from mdn import (
    MixtureDensityNetworkClassifier_wo_T
)

np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
print ("PyTorch version:[%s]."%(torch.__version__))

def main(args):
    
    output_pth = f'./LPM/result/{args.name}'
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
        
    dc = get_ddpm_constants(
        schedule_name = 'cosine', # 'linear', 'cosine'
        T             = 1000,
        np_type       = np.float32,
    )

    # Instantiate U-net
    device = 'cuda' # mps, cpu
    model = MixtureDensityNetworkClassifier_wo_T(
        name       = 'mdn',
        x_dim      = 30,
        y_dim      = 1,
        length     = 30,
        k          = 10,
        h_dim_list = [64,64],
        actv       = nn.SiLU(),
        sig_max    = 1.0,
        mu_min     = -3.0,
        mu_max     = +3.0,
        p_drop     = 0.0,
        use_bn     = False,
        device               = device,
    ) # input:[B x C x L] => output:[B x C x L]
    print ("Ready.")


    train_data = np.load('/nas/home/drizzle0171/motion-generation/MotionGPDiffusion/LPM/train_less.npy', allow_pickle=True)
    train_data = train_data.item()
    train_times = train_data['times']
    train_x_0 = train_data['x_0']
    train_label = train_data['real_param']

    val_data = np.load('/nas/home/drizzle0171/motion-generation/MotionGPDiffusion/LPM/val_less.npy', allow_pickle=True)
    val_data = val_data.item()
    val_times = val_data['times']
    val_x_0 = val_data['x_0']
    val_label = val_data['real_param']

    train_nomalized_data = (train_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())
    val_nomalized_data = (val_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())

    print(train_x_0.shape)

    cls_value = torch.Tensor([0.03, 0.12,
                            0.21, 0.3,
                            0.39, 0.48,
                            0.57, 0.66,
                            0.8, 1.0]).to(device)


    max_iter    = int(2e4)
    batch_size  = 64
    print_every = 1e3
    eval_every  = 1e3

    model.train()
    optm = torch.optim.AdamW(params=model.parameters(),lr=args.lr,weight_decay=args.wd)
    if args.schd == 'exp':
        schd = torch.optim.lr_scheduler.ExponentialLR(optimizer=optm,gamma=0.99998)
    else:
        schd = torch.optim.lr_scheduler.CosineAnnealingLR(optm, T_max=50, eta_min=0)

    min_loss = 1_000_000
    for it in range(max_iter):
    # Zero gradient
        model.train()
        optm.zero_grad()

        # Get batch
        idx = np.random.choice(train_x_0.shape[0],batch_size)
        x_0_batch = train_nomalized_data[idx,:,:] # [B x C x L]
        label = train_label[idx].type(torch.LongTensor).to(device) # [B x C]

        # Sample time steps
        step_batch = torch.randint(0, dc['T'],(batch_size,),device=device).long() # [B]

        # Forward diffusion sampling
        x_t_batch,noise = forward_sample(x_0_batch,step_batch,dc) # [B x C x L]
                            
        # Noise prediction
        # output = model(x_t_batch, step_batch) # [B x C x L]
        # output = model(x_0_batch, step_batch) # [B x C x L]
        gmm_out = model(x_t_batch, label) # [B x C x L]
        
        label_float = label.float()
        # Compute error
        loss = torch.mean(gmm_out['nlls']) + 0.1*F.mse_loss(label_float, gmm_out['argmax_mu'])
        
        if args.wandb:
            wandb.log({'Train loss (Cross Entropy)': loss})

        # # Update
        loss.backward()
        optm.step()
        schd.step()

        # Print
        if (it%print_every) == 0 or it == (max_iter-1):
            print ("it:[%7d][%.1f]%% loss:[%.4f]"%(it,100*it/max_iter,loss.item()))
            
        # Evaluate
        if (it%eval_every) == 0 or it == (max_iter-1):
            model.eval()
            with torch.no_grad():
                step = torch.randint(0, dc['T'], (1000, ), device=device).long()
                x_t, noise = forward_sample(val_nomalized_data, step, dc)
                val_label = val_label.float().to(device)
                gmm_out = model(x_t, val_label) # [B x 1]

                # accuracy
                label_batch = cls_value.repeat(len(x_t), 1) # [B x 10]
                output_batch = gmm_out['argmax_mu'].repeat(1, 10) # [B x 10]
                arg_loss = (label_batch-output_batch)**2 # [B x 10]
                min_idx = torch.argmin(arg_loss, dim=1, keepdim=True) # [B x 1]
                pred_batch = torch.gather(label_batch, 1, min_idx).squeeze() # [B x 1]
                label = cls_value[val_label.type(torch.LongTensor).to(device)]
                
                import ipdb;ipdb.set_trace()
                loss = torch.mean(gmm_out['nlls']) + 0.1*F.mse_loss(val_label, gmm_out['argmax_mu'])
                acc = torch.sum(pred_batch == label) / len(val_label) * 100
                
                if args.wandb:
                    wandb.log({'Val loss (Cross Entropy)': loss})
                    wandb.log({'Accuracy': acc})
                print ("it:[%7d][%.1f]%% loss:[%.4f] accuracy:[%.2f%%]"%(it,100*it/max_iter,loss.item(), acc.item()))
                
                if loss < min_loss:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optm.state_dict(),
                        'loss': loss
                    }, os.path.join(output_pth, f'ckpt_{it}_{acc}.pt'))
                    print('Checkpoint save')
                    min_loss = loss
    print ("Done.")

if __name__ == "__main__":
    def fix_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    fix_seed(2023)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="HDM 1D")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--true", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--name", type=str, default='ds100_len01')
    parser.add_argument("--model", type=str, default='unet')
    parser.add_argument("--d_type", type=str, default='gp2')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--schd", type=str, default='cs')
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--len_param", type=str, default='0.001, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0')
    parser.add_argument("--channel", type=str, default='1,2,2,4,8')
    parser.add_argument("--rate", type=str, default='1,2,2,2,2')
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project='hdm-1d', name=args.name)

    main(args)