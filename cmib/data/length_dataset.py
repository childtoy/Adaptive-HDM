from cmib.lafan1 import extract, utils
import numpy as np
import pickle as pkl
import os
import torch 
from scipy.spatial import distance
from LPM.model import LengthUNet
def k_se(x1, x2, gain=1.0, hyp_len=1.0):

    D_sq = distance.cdist(x1/hyp_len, x2/hyp_len, 'sqeuclidean')
    K = gain * np.exp(-D_sq)
    return K

def main():
    num_data = 60 # number of frames
    fps = 30
    save_path = "./dataset/LAFAN/full_dataset.pkl"
    device = 'cuda:0'
    # load length prediction module 
    model_path = ''
    model = LengthUNet()
    model.load_state_dict()
     
    with open('lafan dataset(pkl_dataset.py) path ', 'rb') as f :
        data = pkl.load(f)
        
    rot_data = data['input_rnorm'][:,:,:-1,:]

    rot_data = torch.Tensor(rot_data).to(device).permute(0,2,3,1).reshape(-1,60).unsqueeze(1)
    model.eval()
    with torch.no_grad():
       pred = model(rot_data)
    #
    #
    # pred_array # (40296, 22, 6)
    lens_array = torch.cat([pred_array, torch.ones([40296,1,6])*0.03], dims=1)
    
    lens_array = lens_array.cpu().numpy()
    t_data = np.linspace(start=0.0, stop=(num_data/fps), num=num_data).reshape((-1,1)) 
    n_sample, num_joint, num_dim = lens_array.shape
    decom_K = np.zeros(shape=(n_sample, num_joint, num_dim,num_data,num_data)) # [N x J x D x L x L]    
    for i in range(n_sample):
        for j in range(num_joint):
            for k in range(num_dim):
                hyp_len = lens_array[i,j,k]
                K = k_se(x1=t_data, x2=t_data, gain=0.1, hyp_len=hyp_len)
                K = K + 1e-6*np.eye(num_data,num_data)
                U, V = np.linalg.eigh(K,UPLO='L')
                decom_K[i,j,k,:,:] = V @ np.diag(np.sqrt(U)) # [L x L]
            
    
    data['decom_K'] = decom_K 
    data['lens_array'] = lens_array
    with open(save_path, 'wb') as f : 
        pkl.dump(data, f)

if __name__ == "__main__":
    main()


##
 