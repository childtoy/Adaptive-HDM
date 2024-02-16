import os
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torchvision import datasets,transforms
from experiment_1d.util import gp_sampler


def get_1d_data(
    n_traj    = 1000, # train per 1,000, val per 100
    L         = 30,
    device    = 'cpu',
    seed      = 1,
    ):
    
    # real_param = {}
    
    if seed is not None:
        np.random.seed(seed=seed)
    times = np.linspace(start=0.0,stop=1.0,num=L).reshape((-1,1)) # [L x 1]
    
    # hyp_len_candidate = np.arange(0.03, 1.0, 0.03).tolist()
    # hyp_len_candidate = hyp_len_candidate[:-1] + [1.0] # remove 0.99 + add 1.0
    hyp_len_candidate = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18,
                        0.21, 0.24, 0.27, 0.3, 0.33, 0.36,
                        0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 
                        0.57, 0.6, 0.63, 0.66, 0.69, 0.72,
                        0.75, 0.78, 0.81, 0.84, 0.9, 1.0]

    traj_list = []
    label = []
    
    for len_param in hyp_len_candidate:
        traj_list_len = np.zeros((n_traj,L))
        for i_idx in range(n_traj):
            label.append(hyp_len_candidate.index(len_param))
            traj_list_len[i_idx,:] = gp_sampler(
                times    = times,
                hyp_gain = 2.0,
                hyp_len  = len_param,
                meas_std = 1e-8,
                n_traj   = 1
            ).reshape(-1)
            # real_param[traj_list_len[i_idx,:]] = len_param
        traj_list.append(traj_list_len)
    traj_np = np.array(traj_list).reshape(-1, 30)
    # np.random.shuffle(traj_np)
    traj = th.from_numpy(
        traj_np
    ).to(th.float32).to(device) # [n_traj x L]
    
    # label = []
    # for i in traj:
    #     label.append(real_param[i])
    
    label = th.Tensor(label)
    label = label.to(th.float32).to(device)
    
    x_0 = traj[:, None, :]
    
    return times, x_0, label