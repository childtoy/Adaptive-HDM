import os
import math
import wandb
import random
import argparse
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import R2Score

from dataset import get_1d_data
from model import (
    LengthPredctionUnet,
)

def main(args):
    starttime = dt.datetime.now()
    print(f'> > Train START {starttime.hour}:{starttime.minute}:{starttime.second}')

    output_pth = f'./result/{args.name}'
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    
    if torch.cuda.is_available():
        device =  'cuda' # mps, cpu
    
    EMD = args.feature
    BLOCK = args.block
    CHANNEL = list(map(int, args.channel.split(',')))
    RATE = list(map(int, args.rate.split(',')))
    ZERO = 1 if args.zero_padding != 'full' else 0
    # ZERO = 0
    model = LengthPredctionUnet(
        name                 = 'unet',
        length               = 60,
        dims                 = 1,
        n_in_channels        = 1,
        n_base_channels      = 128,
        n_emb_dim            = EMD,
        n_cond_dim           = ZERO,
        n_time_dim           = 0,
        n_enc_blocks         = BLOCK, # number of encoder blocks
        n_groups             = 16, # group norm paramter
        n_heads              = 4, # number of heads in QKV attention
        actv                 = nn.SiLU(),
        kernel_size          = 3, # kernel size (3)
        padding              = 1, # padding size (1)
        skip_connection      = True, # additional skip connection
        chnnel_multiples     = CHANNEL,
        updown_rates         = RATE,
        use_scale_shift_norm = True,
        device               = device,
    ) # input:[B x C x L] => output:[B x C x L]
        
    print ("Ready.")

    if args.test == True and args.train == False:
        model.load_state_dict(torch.load(args.ckpt)['model_state_dict'])
    
    # ####################################################
    # times, train_x_0, train_label = get_1d_data(
    #     n_traj    = 200, # train per 1,000, val per 100
    #     L         = 196,
    #     device    = 'cuda:0',
    #     seed      = 1,
    #     )
    
    # train_data = {}
    # train_data['x_0'] = train_x_0
    # train_data['real_param'] = train_label
    # np.save('./data/train_ds100.npy', train_data)
    
    # times, val_x_0, val_label = get_1d_data(
    #     n_traj    = 20, # train per 1,000, val per 100
    #     L         = 196,
    #     device    = 'cuda:0',
    #     seed      = 2,
    #     )
    
    # val_data = {}
    # val_data['x_0'] = val_x_0
    # val_data['real_param'] = val_label
    # np.save('./data/val_ds100.npy', val_data)

    # exit()
    # ####################################################

    train_data = np.load(f'./data/train_ds{args.ds}.npy', allow_pickle=True)
    train_data = train_data.item()
    train_x_0 = train_data['x_0']
    train_label = train_data['real_param']
            
    val_data = np.load(f'./data/val_ds{args.ds}.npy', allow_pickle=True)
    val_data = val_data.item()
    val_x_0 = val_data['x_0']
    val_label = val_data['real_param']

    # 0,1,2,3,4,6,9
    # train_nomalized_data = (train_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())
    # val_normalized_data = (val_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())
    train_nomalized_data = train_x_0
    val_normalized_data = val_x_0

    criterion = torch.nn.MSELoss()
    r2 = R2Score().to(device)

    if args.zero_padding == 'm':
        zero_rate = [0.3, 0.4, 0.5]
    else:
        zero_rate = [0.5]
            
    if args.train:
        max_iter    = int(2.5e5)
        batch_size  = 128
        print_every = 1e3
        eval_every  = 1e3

        # Loop
        model.to(device)
        model.train()
        optm = torch.optim.AdamW(params=model.parameters(),lr=args.lr,weight_decay=args.lr)
        schd = torch.optim.lr_scheduler.CosineAnnealingLR(optm, T_max=max_iter, eta_min=0)
            
        min_loss = 1_000_000
        for it in range(max_iter):
            # Zero gradient
            model.train()
            optm.zero_grad()
            
            # Get batch
            idx = np.random.choice(train_nomalized_data.shape[0],batch_size)
            x_0_batch = train_nomalized_data[idx,:,:] # [B x C x L]
            label = train_label[idx].to(device) # [B x C]
            
            true_length_batch = None
            # Zero padding
            if args.zero_padding != 'full':
                zero_tensor = torch.zeros_like(x_0_batch)
                zero = np.random.choice(zero_rate, int(batch_size/2))
                zero_idx = torch.from_numpy(x_0_batch.shape[2] * zero).long().tolist()
                for i in range(len(zero_idx)):
                    x_0_batch[int(batch_size/2)+i:, :, zero_idx[i]:] = zero_tensor[int(batch_size/2)+i, :, zero_idx[i]:]
                true_length_batch = torch.cat([torch.Tensor([196] * int(batch_size/2)), torch.from_numpy(x_0_batch.shape[2] - zero*x_0_batch.shape[2])], dim=0).float().to(device)
            
            # Class prediction
            output = model(x_0_batch, c=true_length_batch) # [B x C x L]

            # Compute error
            loss = criterion(output.squeeze(), label)
            
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
                model.eval()
                with torch.no_grad():
                    val_batch_size = val_normalized_data.shape[0]
                    true_length_batch = None
                    if args.zero_padding != 'full':
                        true_length_batch = torch.Tensor([196] * val_batch_size).cuda()
                        # zero_tensor = torch.zeros_like(val_normalized_data)
                        # zero = np.random.choice(zero_rate, int(val_batch_size/2))
                        # zero_idx = torch.from_numpy(val_normalized_data.shape[2] * zero).long().tolist()
                        # for i in range(len(zero_idx)):
                        #     val_normalized_data[int(val_batch_size/2)+i:, :, zero_idx[i]:] = zero_tensor[int(val_batch_size/2)+i, :, zero_idx[i]:]
                        # true_length_batch = torch.cat([torch.Tensor([196] * int(val_batch_size/2)), torch.from_numpy(val_normalized_data.shape[2] - zero*val_normalized_data.shape[2])], dim=0).float().to(device)
                    val_label = val_label.to(device)
                    output = model(val_x_0[:,:,:], c=true_length_batch)
                    loss = criterion(output.squeeze(), val_label)
                    r2.update(output.squeeze(), val_label)
                    r2_score = r2.compute().mean()

                    if args.wandb:
                        wandb.log({'Val loss (Cross Entropy)': loss})
                        wandb.log({'Val R2': r2_score})
                        
                    print ("it:[%7d][%.1f]%% loss:[%.4f] r2:[%.4f]"%(it,100*it/max_iter,loss.item(),r2_score.item()))
                    
                    if loss < min_loss:
                        loss_r = round(loss.item(), 5)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optm.state_dict(),
                            'loss': loss
                        }, os.path.join(output_pth, f'ckpt_{it}_{loss}.pt'))
                        print('Checkpoint save')
                        min_loss = loss
        print ("Done.")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optm.state_dict(),
            'loss': loss
        }, os.path.join(output_pth, 'last_ckpt.pt'))
    else:
        model.eval()
        with torch.no_grad():
            # zero-none, zero-one, zero-many
            true_length_batch = None
            
            # if args.zero_padding == 'm':
            #     zero_rate = [0.3, 0.4, 0.5]
            # else:
            #     zero_rate = [0.5]
            
            val_batch_size = val_normalized_data.shape[0]
            idx = np.random.choice(val_normalized_data.shape[0],val_batch_size)
            val_normalized_data = val_normalized_data[idx,:,:]
            if args.zero_padding != 'full':
                true_length_batch = torch.Tensor([196] * val_batch_size).cuda()
                zero_tensor = torch.zeros_like(val_normalized_data)
                zero = np.random.choice(zero_rate, int(val_batch_size))
                zero_idx = torch.from_numpy(val_normalized_data.shape[2] * zero).long().tolist()
                for i in range(len(zero_idx)):
                    val_normalized_data[i:, :, zero_idx[i]:] = zero_tensor[i, :, zero_idx[i]:]
                true_length_batch = torch.cat([torch.from_numpy(val_normalized_data.shape[2] - zero)], dim=0).float().to(device)
            
            # Class prediction
            output = model(val_normalized_data, c=true_length_batch) # [B x C x L]
            val_label = val_label[idx].type(torch.LongTensor).to(device)
            _, pred_idx = torch.max(output.data, 1)
            pred = cls_value[pred_idx]
            label = cls_value[val_label]
            loss = criterion(output, val_label)
            r2_score = r2(output, val_label)
            acc = torch.sum(pred == label) / len(val_label) * 100
            
            print(f'Accuracy: {acc} %')
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
    parser.add_argument("--model", type=str, default='unet')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--zero_padding", type=str, default='full')
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--channel", type=str, default='1,2,2,2,4,4,8')
    parser.add_argument("--rate", type=str, default='1,1,2,1,2,1,2')
    parser.add_argument("--block", type=int, default=7)
    parser.add_argument("--ds", type=int, default=7)
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project='hdm-1d', name=args.name)
    
    main(args)
    
    ### Best accuracy command
    # python -u main.py --name final --train --feature 128 --block 7 --channel '1,2,2,2,4,4,8' --rate '1,1,2,1,2,1,2'
    ###