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
from lpm.dataset import get_1d_data
from lpm.model import (
    LengthPredctionUnet,
)
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main(args):
    
    fix_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    
    model = LengthPredctionUnet(
        name                 = 'unet',
        length               = 60,
        dims                 = 1,
        n_in_channels        = 1,
        n_base_channels      = 128,
        n_emb_dim            = EMD,
        n_cond_dim           = 1,
        n_time_dim           = 0,
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
        model.load_state_dict(torch.load(args.ckpt)['model_state_dict'])

    train_data = np.load('./lpm/data/train_lpm.npy', allow_pickle=True)
    train_data = train_data.item()
    train_x_0 = train_data['x_0']
    train_label = train_data['real_param']
            
    val_data = np.load('./lpm/data/val_lpm.npy', allow_pickle=True)
    val_data = val_data.item()
    val_x_0 = val_data['x_0']
    val_label = val_data['real_param']
    
    train_nomalized_data = (train_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())
    val_nomalized_data = (val_x_0 - train_x_0.mean()) / math.sqrt(train_x_0.var())
    
    cls_value = torch.Tensor(
        [0.033     , 0.14044444, 
        0.24788889, 0.35533333, 
        0.46277778, 0.67766667, 
        1.        ]
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.train:
        max_iter    = int(1e5)
        batch_size  = 128
        print_every = 1e3
        eval_every  = 1e3
        zero_rate   = [0.3, 0.4, 0.5]

        # Loop
        model.to(device)
        model.train()
        optm = torch.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=0.1)

        schd = torch.optim.lr_scheduler.CosineAnnealingLR(optm, T_max=max_iter, eta_min=0)
            
        min_loss = 1_000_000
        for it in range(max_iter):
            # Zero gradient
            model.train()
            optm.zero_grad()
            
            # Get batch
            idx = np.random.choice(train_nomalized_data.shape[0],batch_size)
            x_0_batch = train_nomalized_data[idx,:,:] # [B x C x L]
            label = train_label[idx].type(torch.LongTensor).to(device) # [B x C]
            
            # Zero padding
            zero_tensor = torch.zeros_like(x_0_batch)
            zero = np.random.choice(zero_rate, int(batch_size/2))
            padding_start = torch.from_numpy(x_0_batch.shape[2] * zero).long().tolist()
            padding_idx = np.random.choice(x_0_batch.shape[0],int(batch_size/2))
            true_length_batch = torch.Tensor([x_0_batch.shape[2]] * batch_size).to(device)

            for i, (start, p_idx) in enumerate(zip(padding_start, padding_idx)):
                x_0_batch[p_idx, :, start:] = zero_tensor[p_idx, :, start:]
                true_length_batch[p_idx] = x_0_batch.shape[2] - start
            
            # Class prediction
            output = model(x_0_batch, c=true_length_batch) # [B x C x L]
            
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
                model.eval()
                with torch.no_grad():
                    # batch size for validation set
                    val_batch_size = val_nomalized_data.shape[0]
                    val_label = val_label.type(torch.LongTensor).to(device)
                    
                    # zero padding
                    zero_tensor = torch.zeros_like(val_nomalized_data)
                    zero = np.random.choice(zero_rate, int(val_batch_size/2))
                    padding_start = torch.from_numpy(val_nomalized_data.shape[2] * zero).long().tolist()
                    padding_idx = np.random.choice(val_nomalized_data.shape[0],int(val_batch_size/2))
                    true_length_batch = torch.Tensor([val_nomalized_data.shape[2]] * val_batch_size).to(device)
                    for i, (start, p_idx) in enumerate(zip(padding_start, padding_idx)):
                        val_nomalized_data[p_idx:, :, start:] = zero_tensor[p_idx, :, start:]
                        true_length_batch[p_idx] = val_nomalized_data.shape[2] - start
                    
                    # Calss prediction
                    output = model(val_nomalized_data, c=true_length_batch)
                    _, pred_idx = torch.max(output.data, 1)
                    pred = cls_value[pred_idx]
                    label = cls_value[val_label]
                    
                    # Compute loss
                    loss = criterion(output, val_label)
                    
                    # Extract accuracy
                    acc = torch.sum(pred == label) / len(val_label) * 100
                    
                    if args.wandb:
                        wandb.log({'Val loss (Cross Entropy)': loss})
                        wandb.log({'Accuracy': acc})
                        
                    print ("it:[%7d][%.1f]%% loss:[%.4f] accuracy:[%.2f%%]"%(it,100*it/max_iter,loss.item(), acc.item()))
                    
                    if loss < min_loss:
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
            zero_rate = [0.3, 0.4, 0.5]

            val_batch_size = val_nomalized_data.shape[0]
            val_label = val_label.type(torch.LongTensor).to(device)
            
            true_length_batch = torch.Tensor([val_nomalized_data.shape[2]] * val_batch_size).to(device)
            output = model(val_nomalized_data, c=true_length_batch) # [B x C x L]
            _, pred_idx = torch.max(output.data, 1)
            pred = cls_value[pred_idx]
            label = cls_value[val_label]
            acc_wo_zero = torch.sum(pred == label) / len(val_label) * 100
            import ipdb;ipdb.set_trace()
            cm = confusion_matrix(val_label.detach().cpu().numpy(), pred_idx.detach().cpu().numpy(), normalize='true')
            sns.heatmap(cm, annot=True, cmap='coolwarm')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(list(range(7)), ['0.03', '0.14', '0.24', '0.35', '0.46', '0.67', '1.00'])
            plt.yticks(list(range(7)), ['0.03', '0.14', '0.24', '0.35', '0.46', '0.67', '1.00'])
            plt.savefig(f'./lpm/confusion_matrix_wo_zero.png'); plt.clf()
            plt.savefig(f'./lpm/confusion_matrix_wo_zero.pdf'); plt.clf()
            print(f' * Accuracy w/o Zero padding: {acc_wo_zero} %')
            
            zero_tensor = torch.zeros_like(val_nomalized_data)
            zero = np.random.choice(zero_rate, val_batch_size)
            padding_start = torch.from_numpy(val_nomalized_data.shape[2] * zero).long().tolist()
            padding_idx = np.random.choice(val_nomalized_data.shape[0],val_batch_size)
            true_length_batch = torch.Tensor([val_nomalized_data.shape[2]] * val_batch_size).to(device)
            for i, (start, p_idx) in enumerate(zip(padding_start, padding_idx)):
                val_nomalized_data[p_idx:, :, start:] = zero_tensor[p_idx, :, start:]
                true_length_batch[p_idx] = val_nomalized_data.shape[2] - start
            output = model(val_nomalized_data, c=true_length_batch) # [B x C x L]
            _, pred_idx = torch.max(output.data, 1)
            pred = cls_value[pred_idx]
            label = cls_value[val_label]
            acc_w_zero = torch.sum(pred == label) / len(val_label) * 100
            cm = confusion_matrix(val_label.detach().cpu().numpy(), pred_idx.detach().cpu().numpy(), normalize='true')
            sns.heatmap(cm, annot=True, cmap='coolwarm')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(list(range(7)), ['0.03', '0.14', '0.24', '0.35', '0.46', '0.67', '1.00'])
            plt.yticks(list(range(7)), ['0.03', '0.14', '0.24', '0.35', '0.46', '0.67', '1.00'])
            plt.savefig(f'./lpm/confusion_matrix_w_zero.png')
            plt.savefig(f'./lpm/confusion_matrix_w_zero.pdf')
            print(f' * Accuracy w/ Zero padding: {acc_w_zero} %')
            
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
    
    parser = argparse.ArgumentParser(description="HDM 1D")
    parser.add_argument("--name", type=str, default='lpm')
    parser.add_argument("--model", type=str, default='unet')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--feature", type=int, default=128)
    parser.add_argument("--channel", type=str, default='1,2,2,2,4,4,8')
    parser.add_argument("--rate", type=str, default='1,1,2,1,2,1,2')
    parser.add_argument("--block", type=int, default=7)
    args = parser.parse_args()
    
    # if args.wandb:
    #     wandb.init(project='hdm-1d', name=args.name)
    
    main(args)
    
    ### Best accuracy command
    # python -u main.py --name final --train
    ###