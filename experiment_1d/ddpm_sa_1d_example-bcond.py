import os
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from util import (
    print_model_parameters,
    gp_sampler,
    get_torch_size_string,
    plot_1xN_torch_traj_tensor,
    periodic_step,
    plot_ddpm_1d_result,
)
from dataset_sa import get_1d_training_data
from diffusion import (
    get_ddpm_constants,
    plot_ddpm_constants,
    DiffusionUNetBCond,
    forward_sample,
    eval_ddpm_1d_bcond,
)
from dataset import mnist
np.set_printoptions(precision=3)
th.set_printoptions(precision=3)
print ("PyTorch version:[%s]."%(th.__version__))

dc = get_ddpm_constants(
    schedule_name = 'cosine', # 'linear', 'cosine'
    T             = 1000,
    np_type       = np.float32,
)
device = 'cuda:0' # mps, cpu
print ("device:[%s]"%(device))
# for k_idx,key in enumerate(dc.keys()): print ("[%2d] key:[%s]"%(k_idx,key))
# plot_ddpm_constants(dc)
times,x_0, hyp_lens, traj_split, label_split = get_1d_training_data(
    traj_type = 'gp', # gp / step / step2
    n_traj    = 1,
    L         = 128,
    device    = device,
    seed      = 0,
    plot_data = True,
    verbose   = True,
    split = 4, 
    ) # [N x 1 x 128]

traj_split = traj_split.unsqueeze(1)
traj_split.shape

# Instantiate U-net
model = DiffusionUNetBCond(
    name                 = 'unet',
    dims                 = 1,
    n_in_channels        = 1,
    n_base_channels      = 64,
    n_emb_dim            = 128,
    n_bcond_dim           = 1,
    n_enc_blocks         = 4, # number of encoder blocks
    n_dec_blocks         = 4, # number of decoder blocks
    n_groups             = 16, # group norm paramter
    n_heads              = 4, # number of heads in QKV attention
    actv                 = nn.SiLU(),
    kernel_size          = 3, # kernel size (3)
    padding              = 1, # padding size (1)
    use_attention        = False,
    skip_connection      = True, # additional skip connection
    chnnel_multiples     = (1,2,4,8),
    updown_rates         = (1,2,2,2),
    use_scale_shift_norm = True,
    device               = device,
) # input:[B x C x L] => output:[B x C x L]

# Configuration
max_iter    = int(5e5)
batch_size  = 128
print_every = 1e3
eval_every  = 1e3

# Loop
model.train()
optm = th.optim.AdamW(params=model.parameters(),lr=1e-4,weight_decay=0.0)
schd = th.optim.lr_scheduler.ExponentialLR(optimizer=optm,gamma=0.99998)
outdir = './save/bcond_result'
if not os.path.exists(outdir): os.makedirs(outdir)
for it in range(max_iter):
    # Zero gradient
    optm.zero_grad()
    # Get batch
    idx = np.random.choice(x_0.shape[0],batch_size)
    x_0_batch = traj_split[idx,:,:] # [B x C x L]
    cond_batch = label_split[idx,:].float().unsqueeze(1)
    # Sample time steps
    step_batch = th.randint(0, dc['T'],(batch_size,),device=device).long() # [B]
    # Forward diffusion sampling
    x_t_batch,noise = forward_sample(x_0_batch,step_batch,dc) # [B x C x L]
    # Noise prediction
    noise_pred,_ = model(x_t_batch,step_batch,cond_batch) # [B x C x L]
    # Compute error
    loss = F.smooth_l1_loss(noise,noise_pred,beta=0.1)
    # Update
    loss.backward()
    optm.step()
    schd.step()
    # Print
    if (it%print_every) == 0 or it == (max_iter-1):
        print ("it:[%7d][%.1f]%% loss:[%.4f]"%(it,100*it/max_iter,loss.item()))
    # Evaluate
    if (it%eval_every) == 0 or it == (max_iter-1):
        th.save(model.state_dict(), outdir+'/bcond_'+str(it+1)+'.pt') 

print ("Training done.")

# Evaluate
n_sample = 100

step_list_to_append = np.linspace(0,999,10).astype(np.int64)
cond_label = th.repeat_interleave(label_split,25,dim=0).float().unsqueeze(1)
x_t_list = eval_ddpm_1d_bcond(model,dc,n_sample,traj_split,step_list_to_append,device, cond=cond_label)
print ("Sampling [%d] trajectories done."%(n_sample))
# for _ in range(100):
#     plot_ddpm_1d_result(
#         times=times[:32],x_data=traj_split,step_list=step_list_to_append,x_t_list=x_t_list,
#         plot_ancestral_sampling=False,plot_one_sample=True,
#         lw_gt=2,lw_sample=1/2,lc_gt=(0,1,0,0.3),lc_sample='k',
#         ls_gt='-',ls_sample='--',ylim=(-4,+4),figsize=(5,3))

from matplotlib import pyplot as plt
fig, axs = plt.subplots(2,2)
# fig.tight_layout()

axs[0,0].plot(times[:32], traj_split[0,0].cpu().numpy(),label='0')
axs[0,0].plot(times[:32], traj_split[1,0].cpu().numpy(),label='1')
axs[0,0].plot(times[:32], traj_split[2,0].cpu().numpy(),label='2')
axs[0,0].plot(times[:32], traj_split[3,0].cpu().numpy(),label='3')
for i in range(20):
    axs[0,0].plot(times[:32], x_t_list[0][i,0].cpu().numpy(),linestyle='--')
# axs[0,0].set_title('Groundtruth and Generated trajectory given label 0')

axs[0,1].plot(times[:32], traj_split[0,0].cpu().numpy())
axs[0,1].plot(times[:32], traj_split[1,0].cpu().numpy())
axs[0,1].plot(times[:32], traj_split[2,0].cpu().numpy())
axs[0,1].plot(times[:32], traj_split[3,0].cpu().numpy())
for i in range(20):
    axs[0,1].plot(times[:32], x_t_list[0][25+i,0].cpu().numpy(),linestyle='--')
# axs[0,1].set_title('Groundtruth and Generated trajectory given label 1')

axs[1,0].plot(times[:32], traj_split[0,0].cpu().numpy())
axs[1,0].plot(times[:32], traj_split[1,0].cpu().numpy())
axs[1,0].plot(times[:32], traj_split[2,0].cpu().numpy())
axs[1,0].plot(times[:32], traj_split[3,0].cpu().numpy())
for i in range(20):
    axs[1,0].plot(times[:32], x_t_list[0][50+i,0].cpu().numpy(),linestyle='--')
# axs[1,0].set_title('Groundtruth and Generated trajectory given label 2')

axs[1,1].plot(times[:32], traj_split[0,0].cpu().numpy())
axs[1,1].plot(times[:32], traj_split[1,0].cpu().numpy())
axs[1,1].plot(times[:32], traj_split[2,0].cpu().numpy())
axs[1,1].plot(times[:32], traj_split[3,0].cpu().numpy())
for i in range(20):
    axs[1,1].plot(times[:32], x_t_list[0][75+i,0].cpu().numpy(),linestyle='--')
# axs[1,1].set_title('Groundtruth and Generated trajectory given label 3')
fig.suptitle('Groundtruth and Each Generated trajectory given label')
fig.legend()
fig.savefig('result_bcond.png')