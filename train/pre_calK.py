# This code is based on https://github.com/openai/guided-diffusion
"""
Save GP kernel parameters for correlated noise
"""

import os
import json
import torch
import numpy as np 
import pickle as pkl
from scipy.spatial import distance

def k_se(x1, x2, gain=1.0, hyp_len=1.0):

    D_sq = distance.cdist(x1/hyp_len, x2/hyp_len, 'sqeuclidean')
    K = gain * np.exp(-D_sq)
    return K

def main():
    num_data = 196 # number of frames
    fps = 20
    num_dim = 263 # data rep dimension
    # num_dim = 2 # data rep dimension
    num_len = 10 # number of length parameters
    save_path = './HumanML3D_K_param_data'+str(num_data)+'_fps'+str(fps)+'_dim'+str(num_dim)+'_len'+str(num_len)+'.pkl'
    
    t_data = np.linspace(start=0.0, stop=(num_data/fps), num=num_data).reshape((-1,1)) # num frames : 128, fps : 20 
    # lens_array = np.linspace(0.01,num_data/fps,num_len) # GP kernel length param array
    lens_array =  np.array([0.03, 0.12, 0.21, 0.3, 0.39, 0.48, 0.57, 0.66,0.8, 1.0])
    decom_K = np.zeros(shape=(len(lens_array), num_data,num_data)) # [D x L x L]    
    for len_idx in range(len(lens_array)):        
        # for d_idx in range(num_dim):
        hyp_len = lens_array[len_idx]
        K = k_se(x1=t_data, x2=t_data, gain=0.1, hyp_len=hyp_len)
        K = K + 1e-6*np.eye(num_data,num_data)
        U, V = np.linalg.eigh(K,UPLO='L')
        decom_K[len_idx,:,:] = V @ np.diag(np.sqrt(U)) # [L x L]
        
    template_decom_K = np.zeros(shape=(num_dim, num_data,num_data)) # [D x L x L]    
    for dim_idx in range(num_dim):
        # for d_idx in range(num_dim):
        hyp_len = 0.03
        K = k_se(x1=t_data, x2=t_data, gain=0.1, hyp_len=hyp_len)
        K = K + 1e-6*np.eye(num_data,num_data)
        U, V = np.linalg.eigh(K,UPLO='L')
        template_decom_K[dim_idx,:,:] = V @ np.diag(np.sqrt(U)) # [L x L]
    
    data = {'template': template_decom_K, 'K_param' : decom_K, 'len_param' : lens_array}
    with open(save_path, 'wb') as f : 
        pkl.dump(data, f)

if __name__ == "__main__":
    main()
