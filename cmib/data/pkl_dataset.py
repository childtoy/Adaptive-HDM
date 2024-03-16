from torch.utils.data import Dataset
from cmib.lafan1 import extract, utils
import numpy as np
import pickle
import os
from data_loaders.humanml.common.quaternion import quaternion_to_matrix_np
from cmib.data.utils import flip_bvh


# for viz 
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, sk_parents, sk_skeleton_part)
from data_loaders.humanml.common.quaternion import cont6d_to_quat
import os 
import pickle as pkl
import numpy as np 
import torch 
import sys 


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


class LAFAN1Dataset():
    def __init__(
        self,
        lafan_path: str,
        processed_data_dir: str,
        train: bool,
        window: int = 70,
    ):
        self.lafan_path = lafan_path

        self.train = train
        self.actors = (
            ["subject1", "subject2", "subject3", "subject4"] if train else ["subject5"]
        )
        self.window = window
        self.offset = 20 if self.train else 40
        pickle_name = "lafan_"+str(window-10)+"_train_data.pkl" if train else "lafan_"+str(window-10)+"_test_data.pkl"
        flip_bvh(self.lafan_path, skip='subject5')

        self.data = self.load_lafan()  # Call this last
        os.makedirs(processed_data_dir, exist_ok=True)
        with open(os.path.join(processed_data_dir, pickle_name), "wb") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def load_lafan(self):
        offset = sk_offsets 
        skeleton_mocap = Skeleton(offsets=offset, parents=sk_parents, device=torch.device('cpu'))
        skeleton_mocap.remove_joints(sk_joints_to_remove)

        skeleton_mocap._parents
        X, Q, parents = extract.get_lafan1_set(
            self.lafan_path, self.actors, self.window, self.offset, self.train,
        )
        _, global_pos = utils.quat_fk(Q, X, parents)
        
        input_data = {}
        input_data["rot_6d"] = quaternion_to_cont6d_np(Q)[:, 9:self.window-1, :, :]
        input_data["root_p"] = global_pos[:, 9:self.window-1, 0, :]
        N, L, J, D = global_pos.shape


        mean_root = global_pos[:, 9:self.window-1, 0, :].mean(axis=0)
        std_root = global_pos[:, 9:self.window-1, 0, :].std(axis=0)
        std_root_xy =  std_root[:,[0,2]].mean()
        std_root_z = std_root[:,1].mean()
        
        std_root_xzy = np.array([std_root_xy, std_root_z, std_root_xy])
        mean_rot = input_data["rot_6d"].mean(axis=0)
        std_rot = input_data["rot_6d"].std(axis=0)
        std_rot = std_rot.mean()
        input_data["mean_root"] = mean_root
        input_data["std_root"] = std_root_xzy
        input_data["mean_rot"] = mean_rot
        input_data["std_rot"] = std_rot
        root_norm = (global_pos[:, 9:self.window-1, 0, :] - np.expand_dims(mean_root,axis=0))/std_root_xzy
        rot_norm = (input_data["rot_6d"] - mean_rot) / std_rot
        padded_root = np.concatenate([global_pos[:, 9:self.window-1, 0:1, :], np.zeros([N,self.window-10,1,3])], axis=-1)
        padded_root_norm = np.concatenate([np.expand_dims(root_norm, axis=2), np.zeros([N,self.window-10,1,3])], axis=-1)
        input_data["input_data"] = np.concatenate([input_data["rot_6d"], padded_root], axis=2)
        input_data["input_norm"] = np.concatenate([rot_norm, padded_root_norm], axis=2)

        # sample = input_data["input_data"]
        # print(sample.shape)
        # # sample = input_data
        # root_pos = sample[:,:,-1,:3]
        # rot_6d = sample[:,:,:-1,:]
        # # print(rot_6d.shape)
        # rot_quat = cont6d_to_quat(torch.Tensor(rot_6d))
        # print(rot_quat.shape,'rotquat')
        # # # rot_quat = rot_quat.reshape(-1,60,22,4)
        # local_q_normalized = torch.nn.functional.normalize(rot_quat), p=2.0, dim=-1)
        # print('local_q_normalized', local_q_normalized.shape)
        # print('root_pos', root_pos.shape)
        # global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, torch.Tensor(root_pos))
        # print(global_pos.shape)
        
        # caption = 'unconditioned_LAFAN'
        # length = 60
        # # motion = global_pos[0].cpu().numpy().transpose(2, 0, 1)
        # print(global_pos.shape)
        # motion = global_pos[0].numpy()
        # animation_save_path = os.path.join('./', 'check_train.gif')
        # plot_3d_motion(animation_save_path, sk_skeleton_part, motion, dataset='LAFAN', title=caption, fps=30)
        # sys.exit()
        
        return input_data



print('start make LAFAN dataset')
LAFAN1Dataset(lafan_path="/workspace/workspace/childtoy/lafan1/output/BVH",
              processed_data_dir="dataset/LAFAN/", 
              train=True, 
              window=70)

print('finished')