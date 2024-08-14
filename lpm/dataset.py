import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from lpm.util import gp_sampler
from torch.utils.data import Dataset
    
def get_1d_data(
    n_traj    = 1000, # train per 1,000, val per 100
    L         = 30,
    device    = 'cpu',
    seed      = 1,
    eval      = False
    ):
    
    # real_param = {}
    
    if seed is not None:
        np.random.seed(seed=seed)
    times = np.linspace(start=0.0,stop=2.0,num=L).reshape((-1,1)) # [L x 1]

    traj_list = []
    label = []
    
    # train
    hyp_len_candidate = [0.033     , 0.14044444, 0.24788889, 
                        0.35533333, 0.46277778,
                        0.67766667, 1.        ]
    
    # evaluation
    if eval:
        given_points = np.array(hyp_len_candidate)
        extra_points = np.linspace(0.01, 1, 20-len(given_points)+1)
        hyp_len_candidate = np.unique(np.concatenate((extra_points, given_points)))
        hyp_len_candidate = hyp_len_candidate.tolist()
        
    print('# of length param: ', len(hyp_len_candidate))
    
    for len_param in hyp_len_candidate:
        traj_list_len = np.zeros((n_traj,L))
        for i_idx in range(n_traj):
            label.append(hyp_len_candidate.index(len_param))
            while True:
                sample = gp_sampler(
                    times    = times,
                    hyp_gain = 0.1,
                    hyp_len  = len_param,
                    meas_std = 0e-8,
                    n_traj   = 1
                ).reshape(-1)
                if np.max(sample[0]) < 1 and np.min(sample[0]) > -1:
                    break
            traj_list_len[i_idx,:] = sample
        traj_list.append(traj_list_len)
    traj_np = np.array(traj_list).reshape(-1, L)
    traj = th.from_numpy(
        traj_np
    ).to(th.float32).to(device)
    
    label = th.Tensor(label)
    label = label.to(th.float32).to(device)
    
    x_0 = traj[:, None, :]
    
    return times, x_0, label

if __name__ == '__main__':
    
    # for length prediction module
    os.makedirs('./lpm/data', exist_ok=True)
    
    times, train_x_0, train_label = get_1d_data(
        n_traj    = 1000, # train per 1,000, val per 100
        L         = 196,
        device    = 'cuda:0',
        seed      = 42,
        )

    times, val_x_0, val_label = get_1d_data(
        n_traj    = 100, # train per 1,000, val per 100
        L         = 196,
        device    = 'cuda:0',
        seed      = 42,
        )

    train_data = {}
    train_data['x_0'] = train_x_0
    train_data['real_param'] = train_label
    np.save('./lpm/data/train_lpm.npy', train_data)
    
    val_data = {}
    val_data['x_0'] = val_x_0
    val_data['real_param'] = val_label
    np.save('./lpm/data/val_lpm.npy', val_data)
    
    # for 1d generation
    os.makedirs('./1d-generation/data', exist_ok=True)
    
    times, train_x_0, train_label = get_1d_data(
        n_traj    = 1000, # train per 1,000, val per 100
        L         = 196,
        device    = 'cuda:0',
        seed      = 1234,
        )

    times, val_x_0, val_label = get_1d_data(
        n_traj    = 100, # train per 1,000, val per 100
        L         = 196,
        device    = 'cuda:0',
        seed      = 1234,
        )

    train_data = {}
    train_data['x_0'] = train_x_0
    train_data['times'] = times
    train_data['real_param'] = train_label
    np.save('./1d-generation/data/train_gp.npy', train_data)
    
    val_data = {}
    val_data['x_0'] = val_x_0
    val_data['times'] = times
    val_data['real_param'] = val_label
    np.save('./1d-generation/data/val_gp.npy', val_data)