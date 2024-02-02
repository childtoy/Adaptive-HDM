from data_loaders.humanml.utils.plot_script import plot_3d_motion
from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, sk_parents, sk_skeleton_part)
from data_loaders.humanml.common.quaternion import cont6d_to_quat
import os 
import pickle as pkl
import numpy as np 
import torch 


pkldatafilepath = os.path.join('dataset/LAFAN/', "lafan_60_train_data.pkl")
data = pkl.load(open(pkldatafilepath, "rb"))
device = torch.device('cuda:0') 
offset = sk_offsets 
skeleton_mocap = Skeleton(offsets=offset, parents=sk_parents, device=device)
skeleton_mocap.remove_joints(sk_joints_to_remove)

skeleton_mocap._parents

input_data = data["input_data"][0].astype(np.float32)
input_data = torch.Tensor(input_data).unsqueeze(0).to(device)
print(input_data.shape)
# sample = input_data.reshape(-1,23,6,60).permute(0,3,1,2)
sample = input_data
root_pos = sample[:,:,-1,:3]
rot_6d = sample[:,:,:-1,:]
print(rot_6d.shape)
rot_quat = cont6d_to_quat(rot_6d)
# rot_quat = rot_quat.reshape(-1,60,22,4)
local_q_normalized = torch.nn.functional.normalize(rot_quat, p=2.0, dim=-1)
global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, root_pos)
print(global_pos.shape)
caption = 'unconditioned_LAFAN'
length = 60
motion = global_pos[0].cpu().numpy().transpose(2, 0, 1)
animation_save_path = os.path.join('./', 'check_train.gif')
plot_3d_motion(animation_save_path, sk_skeleton_part, motion, dataset='LAFAN', title=caption, fps=30)
