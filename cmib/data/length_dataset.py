from cmib.lafan1 import extract, utils
import numpy as np
import pickle as pkl
import os
import torch 
from scipy.spatial import distance
from LPM.model import LengthPredctionUnet
from tqdm import tqdm
import torch.nn as nn

def k_se(x1, x2, gain=1.0, hyp_len=1.0):

    D_sq = distance.cdist(x1/hyp_len, x2/hyp_len, 'sqeuclidean')
    K = gain * np.exp(-D_sq)
    return K

def main():
    num_data = 60 # number of frames
    fps = 30
    save_path = '/data/full_data_wo_pred_root_norm0.pkl'
    device = 'cuda'
    
    # load length prediction module 
    print('Loading Model')
    model_pth = '/data2/lafan1/ckpt_length60_range1.pt'
    model = LengthPredctionUnet(
            name                 = 'unet',
            length               = 60, 
            dims                 = 1,
            n_in_channels        = 1,
            n_base_channels      = 128,
            n_emb_dim            = 128,
            n_cond_dim           = 0,
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
            device               = 'cuda',
        ) # input:[B x C x L] => output:[B x C x L]
    model.load_state_dict(torch.load(model_pth)['model_state_dict'])
    
    cls_value = torch.Tensor([0.03, 0.12,   
                        0.21, 0.3,
                        0.39, 0.48,
                        0.57, 0.66,
                        0.8, 1.0]).cuda()
    print('Loading Data')
    with open('./dataset/LAFAN/lafan_60_train_data.pkl', 'rb') as f :
        data = pkl.load(f)
    
    print('Predicting length parameter')
    rot_data = data['input_rnorm'][:,:,:-1,:]
    rot_data = (rot_data- np.mean(data['rot_6d'], axis=(0,1)))/np.std(data['rot_6d'], axis=(0))
    rot_data = torch.Tensor(rot_data).reshape(-1, 60, 132) # [40,296, 60, 132]
    model.eval()
    pred_array = []
    num = 200
    
    for i in tqdm(range(0, 40000, num), total=int(40000/num)):
        sample = rot_data[i:i+num, :, :].permute(0, 2, 1).reshape(-1, 60).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(sample)
        _, pred_idx = torch.max(output.data, 1)
        output = pred_idx
        pred = cls_value[pred_idx]
        pred = pred.detach().cpu().numpy()
        sampe = sample.detach().cpu()
        pred_array.append(pred)
    
    # rest of the data
    sample = rot_data[40000:, :, :].permute(0, 2, 1).reshape(-1, 60).unsqueeze(1).to(device)
    with torch.no_grad():
        output = model(sample)
    _, pred_idx = torch.max(output.data, 1)
    output = pred_idx
    pred = cls_value[pred_idx]
    pred = pred.detach().cpu().numpy()
    pred_array.append(pred)
        
    pred_array = torch.from_numpy(np.concatenate(pred_array)).reshape(-1, 22, 6) # [40,296 x 132,] -> [40,296, 22, 6]
    lens_array = torch.cat([pred_array, torch.ones([40296,1,6])*0.03], dim=1) # [40,296, 23, 6]
    
    # print('Calculating Decomposition K')
    lens_array = lens_array.detach().cpu().numpy()
    # t_data = np.linspace(start=0.0, stop=(num_data/fps), num=num_data).reshape((-1,1)) 
    # n_sample, num_joint, num_dim = lens_array.shape
    # decom_K = np.zeros(shape=(n_sample, num_joint, num_dim,num_data,num_data)) # [N x J x D x L x L]    
    # for i in tqdm(range(n_sample)):
    #     for j in range(num_joint):
    #         for k in range(num_dim):
    #             hyp_len = lens_array[i,j,k]
    #             K = k_se(x1=t_data, t_data, gain=0.1, hyp_len=hyp_len)
    #             K = K + 1e-6*np.eye(num_data,num_data)
    #             U, V = np.linalg.eigh(K,UPLO='L')
    #             decom_K[i,j,k,:,:] = V @ np.diag(np.sqrt(U)) # [L x L]
            
    
    # data['decom_K'] = decom_K 
    data['lens_array'] = lens_array
    with open(save_path, 'wb') as f : 
        pkl.dump(data, f)

if __name__ == "__main__":
    main()