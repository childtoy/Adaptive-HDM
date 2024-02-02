import pickle as pkl
import numpy as np
import os
from torch.utils.data import Dataset
import torch 

class LAFAN1Dataset(Dataset):
    dataname = "LAFAN"
    
    def __init__(self, datapath="dataset/LAFAN/", split="train", num_frames=60):
        self.datapath = datapath

        super().__init__()

        pkldatafilepath = os.path.join(datapath, "lafan_60_train_data.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        # 4.3: It contains actions performedby 5 subjects, with Subject 5 used as the test set.
        self.data = data
    def __len__(self):
        return self.data["input_data"].shape[0]

    def __getitem__(self, index):
        # rot_6d = self.data["rot_6d"][index].astype(np.float32)        
        # root_p = self.data["root_p"][index].astype(np.float32)
        # padded_root_p = np.concatenate([root_p, np.zeros([3])]).reshape(-1,1,6)
        input_data = self.data["input_data"][index].astype(np.float32)
        # inp = np.concatenate([rot_6d,padded_root_p], axis=1)        
        inp = torch.Tensor(input_data).reshape(60, 1, -1).permute(2,1,0)
        output  = {'inp': inp}        

        return output
