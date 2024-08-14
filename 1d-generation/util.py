import os
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from scipy.spatial import distance

def kernel_se(x1,x2,hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    D = distance.cdist(x1/hyp['len'],x2/hyp['len'],'sqeuclidean')
    K = hyp['gain']*np.exp(-D)
    return K

def gp_sampler(
    times    = np.linspace(start=0.0,stop=1.0,num=100).reshape((-1,1)), # [L x 1]
    hyp_gain = 1.0,
    hyp_len  = 1.0,
    meas_std = 0e-8,
    n_traj   = 1
    ):
    """ 
        Gaussian process sampling
    """
    if len(times.shape) == 1: times = times.reshape((-1,1))
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':hyp_gain,'len':hyp_len}) # [L x L]
    K_chol = np.linalg.cholesky(K+1e-8*np.eye(L,L)) # [L x L]
    traj = K_chol @ np.random.randn(L,n_traj) # [L x n_traj]
    traj = traj + meas_std*np.random.randn(*traj.shape)
    return traj.T


def plot_ddpm_1d_result(
    times,x_data, step_list,x_t_list,real_param = None,
    plot_ancestral_sampling=True,
    plot_one_sample=False,
    lw_gt=1,lw_sample=1/2,
    ls_gt='-',ls_sample='-',
    lc_gt='b',lc_sample='k',
    ylim=(-4,+4),figsize=(6,3),title_str=None,
    output='./', len_param=0.1
    ):
    """
    :param times: [L x 1] ndarray
    :param x_0: [N x C x L] torch tensor, training data
    :param step_list: [M] ndarray, diffusion steps to append x_t
    :param x_t_list: list of [n_sample x C x L] torch tensors
    """
    # Select the data that matched to the len parameter
    target_value = len_param
    indices = th.nonzero(real_param == target_value).squeeze().detach().cpu()
    x_data = x_data[indices, :, :][:20,:,:]

    x_data_np = x_data.detach().cpu().numpy() # [n_data x C x L]
    n_data = x_data_np.shape[0] # number of GT data
    c_idxs = x_data_np.shape[1] 
    # Plot a seqeunce of ancestral sampling procedure
    if plot_ancestral_sampling:
        plt.figure(figsize=(15,2)); plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
        for i_idx,t in enumerate(step_list):
            plt.subplot(1,len(step_list),i_idx+1)
            x_t = x_t_list[t] # [n_sample x C x L]
            x_t_np = x_t.detach().cpu().numpy() # [n_sample x C x L]
            n_sample = x_t_np.shape[0]
            for i_idx in range(n_data): # GT
                plt.plot(times.flatten(),x_data_np[i_idx,0,:],ls='-',color=lc_gt,lw=lw_gt)
            for i_idx in range(n_sample): # sampled trajectories
                plt.plot(times.flatten(),x_t_np[i_idx,0,:],ls='-',color=lc_sample,lw=lw_sample)
            plt.xlim([0.0,1.0]); plt.ylim(ylim)
            plt.xlabel('Time',fontsize=8); plt.title('Step:[%d]'%(t),fontsize=8)
        plt.tight_layout(); plt.show()
    for c_idx in range(c_idxs) : 
        # Plot generated data
        plt.figure(figsize=figsize) 
        x_0_np = x_t_list[0].detach().cpu().numpy() # [n_sample x C x L]
        for i_idx in range(n_data): # GT
            plt.plot(times.flatten(),x_data_np[i_idx,c_idx,:],ls=ls_gt,color=lc_gt,lw=lw_gt)
        n_sample = x_0_np.shape[0]
        if plot_one_sample:
            i_idx = np.random.randint(low=0,high=n_sample)
            plt.plot(times.flatten(),x_0_np[i_idx,c_idx,:],ls=ls_sample,color=lc_sample,lw=lw_sample)
        else:
            for i_idx in range(n_sample): # sampled trajectories
                plt.plot(times.flatten(),x_0_np[i_idx,c_idx,:],ls=ls_sample,color=lc_sample,lw=lw_sample)
        plt.xlim([0.0,1.0]); plt.ylim(ylim)
        plt.xlabel('Time',fontsize=8)
        # if title_str is None:
        plt.title(f'[{c_idx}] Groundtruth and Generated trajectories_iter:{title_str}',fontsize=10)
        # else:
            # plt.title(f'[{c_idx}] '+title_str,fontsize=10)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(output, f'generated_{len_param}.png'))
    
def get_hbm_M(times,hyp_gain=1.0,hyp_len=0.1,device='cpu'):
    """ 
    Get a matrix M for Hilbert Brownian motion
    :param times: [L x 1] ndarray
    :return: [L x L] torch tensor
    """
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':hyp_gain,'len':hyp_len}) # [L x L]
    K = K + 1e-8*np.eye(L,L)
    U,V = np.linalg.eigh(K,UPLO='L')
    M = V @ np.diag(np.sqrt(U))
    M = th.from_numpy(M).to(th.float32).to(device) # [L x L]
    
    return M

def box_plot(sequence, prompt_pth=None):
    lens_str = ['0.03','0.14','0.24','0.35','0.46', '0.67', '1.00']
    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(sequence)
    plt.xlabel('Length parameter')
    plt.xticks(list(range(1,len(lens_str)+1)),lens_str)
    plt.ylabel('Mean Velocity')
    plt.savefig(f'{prompt_pth}.pdf')
    plt.savefig(f'{prompt_pth}.png')
    