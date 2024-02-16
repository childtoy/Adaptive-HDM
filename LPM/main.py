import sys
sys.path.append('/nas/home/drizzle0171/motion-generation/MotionGPDiffusion')

import os
import wandb
import random
import argparse
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_1d_data
from model import (
    get_ddpm_constants,
    DiffusionUNetLegacy,
    forward_sample,
)

def main(args):
    starttime = dt.datetime.now()
    print(f'> > Train START {starttime.hour}:{starttime.minute}:{starttime.second}')

    dc = get_ddpm_constants(
        schedule_name = 'cosine', # 'linear', 'cosine'
        T             = 1000,
        np_type       = np.float32,
    )

    output_pth = f'./LPM/result/{args.name}'
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    
    if torch.cuda.is_available():
        device =  'cuda' # mps, cpu
    
    EMD = args.feature
    BLOCK = args.block
    CHANNEL = list(map(int, args.channel.split(',')))
    RATE = list(map(int, args.rate.split(',')))
    
    model = DiffusionUNetLegacy(
        name                 = 'unet',
        dims                 = 1,
        n_in_channels        = 1,
        n_base_channels      = 128,
        n_emb_dim            = EMD,
        n_cond_dim           = 0,
        n_enc_blocks         = BLOCK, # number of encoder blocks
        n_groups             = 16, # group norm paramter
        n_heads              = 4, # number of heads in QKV attention
        actv                 = nn.SiLU(),
        kernel_size          = 3, # kernel size (3)
        padding              = 1, # padding size (1)
        use_attention        = False,
        skip_connection      = True, # additional skip connection
        chnnel_multiples     = CHANNEL,
        updown_rates         = RATE,
        use_scale_shift_norm = True,
        device               = device,
    ) # input:[B x C x L] => output:[B x C x L]
        
    print ("Ready.")
    
    if args.test == True and args.train == False:
        model.load_state_dict(torch.load(os.path.join(output_pth, 'ckpt.pt'))['model_state_dict'])
    
    # # Dataset
    # times, x_0, label = get_1d_data(
    #     n_traj    = 100,
    #     L         = 30,
    #     device    = device,
    #     seed      = 0,
    #     )

    # import ipdb;ipdb.set_trace()
    # dic = {'times': times, 'x_0': x_0, 'real_param':label}
    # np.save('./val.npy', dic)
        

    train_data = np.load('./train.npy', allow_pickle=True)
    train_data = train_data.item()
    train_times = train_data['times']
    train_x_0 = train_data['x_0']
    train_label = train_data['real_param']

    val_data = np.load('./val.npy', allow_pickle=True)
    val_data = val_data.item()
    val_times = val_data['times']
    val_x_0 = val_data['x_0']
    val_label = val_data['real_param']
    
    print(train_x_0.shape)
    
    cls_value = torch.Tensor([0.03, 0.06, 0.09, 0.12, 0.15, 0.18,
        0.21, 0.24, 0.27, 0.3, 0.33, 0.36,
        0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 
        0.57, 0.6, 0.63, 0.66, 0.69, 0.72,
        0.75, 0.78, 0.81, 0.84, 0.9, 1.0]).to(device)
    
    if args.train:
        # Configuration
        max_iter    = int(5e5)
        batch_size  = 128
        print_every = 1e3
        eval_every  = 1e3

        # Loop
        model.train()
        optm = torch.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=0.0)
        schd = torch.optim.lr_scheduler.ExponentialLR(optimizer=optm,gamma=0.99998)
        criterion = torch.nn.CrossEntropyLoss()
        
        for it in range(max_iter):
            # Zero gradient
            optm.zero_grad()
            
            # Get batch
            idx = np.random.choice(train_x_0.shape[0],batch_size)
            x_0_batch = train_x_0[idx,:,:] # [B x C x L]
            label = train_label[idx].type(torch.LongTensor).to(device) # [B x C]
            
            # Sample time steps
            step_batch = torch.randint(0, dc['T'],(batch_size,),device=device).long() # [B]
            
            # Forward diffusion sampling
            # x_t_batch,noise = forward_sample(x_0_batch,step_batch,dc) # [B x C x L]
                                
            # Noise prediction
            # output = model(x_t_batch, step_batch) # [B x C x L]
            output = model(x_0_batch) # [B x C x L]
            
            # Compute error
            loss = criterion(output, label)
            if args.wandb:
                wandb.log({'Train loss (Cross Entropy)': loss})
            
            # Update
            loss.backward()
            optm.step()
            schd.step()
            
            # Print
            if (it%print_every) == 0 or it == (max_iter-1):
                print ("it:[%7d][%.1f]%% loss:[%.4f]"%(it,100*it/max_iter,loss.item()))
                
            # Evaluate
            if (it%eval_every) == 0 or it == (max_iter-1):
                with torch.no_grad():
                    step = torch.randint(0, dc['T'], (3000, ), device=device).long()
                    x_t, noise = forward_sample(val_x_0, step, dc)
                    val_label = val_label.type(torch.LongTensor).to(device)
                    output = model(x_t, step)
                    _, pred_idx = torch.max(output.data, 1)
                    pred = cls_value[pred_idx]
                    label = cls_value[val_label]
                    loss = criterion(output, val_label)
                    acc = torch.sum(pred == label) / len(val_label) * 100
                    if args.wandb:
                        wandb.log({'Val loss (Cross Entropy)': loss})
                        wandb.log({'Accuracy': acc})
                    print ("it:[%7d][%.1f]%% loss:[%.4f] accuracy:[%.2f%%]"%(it,100*it/max_iter,loss.item(), acc.item()))
                
        print ("Done.")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optm.state_dict(),
            'loss': loss
        }, os.path.join(output_pth, 'ckpt.pt'))

        endtime = dt.datetime.now()
        elapse = (endtime-starttime).seconds
        print(f'\n> > Train END {endtime.hour}:{endtime.minute}:{endtime.second}')
        
        print(f'Elapsed time: {elapse//3600}:{(elapse%3600)//60}:{(elapse)%60}')

if __name__ =="__main__":
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
    parser.add_argument("--name", type=str, default='lpm')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--channel", type=str, default='1,2,4,8')
    parser.add_argument("--rate", type=str, default='1,2,2,2')
    parser.add_argument("--block", type=int, default=4)
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project='hdm-1d', name=args.name)
    
    main(args)