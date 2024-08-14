import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

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

    
def box_plot(sequence, prompt_pth=None):
    lens_str = ['0.03','0.14','0.24','0.35','0.46', '0.67', '1.00']
    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(sequence)
    plt.xlabel('Length parameter')
    plt.xticks(list(range(1,len(lens_str)+1)),lens_str)
    plt.ylabel('Mean Velocity')
    plt.savefig(f'./lpm/result_box_{prompt_pth}.pdf')
    plt.savefig(f'./lpm/result_box_{prompt_pth}.pdf')
    