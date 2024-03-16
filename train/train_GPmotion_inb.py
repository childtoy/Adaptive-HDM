# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_GPmotion_inb import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from cmib.data.lafan1_dataset import LAFAN1Dataset
from cmib.data.utils import flip_bvh, increment_path, process_seq_names
from cmib.model.network import TransformerModel
from cmib.model.preprocess import (lerp_input_repr, replace_constant,
                                   slerp_input_repr, vectorize_representation)
from cmib.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets, sk_parents)
from data_loaders.humanml.common.quaternion import *
import torch  as th
from torch.utils.data import DataLoader, TensorDataset

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data, dataset = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)
    
    # Load LAFAN Dataset
    # print(f"Horizon: {horizon}")
    # print(f"Horizon with Conditioning: {horizon}")
    # print(f"Interpolation Mode: {opt.interpolation}")
    print("creating model and diffusion...")
    model, diffusion, len_model = create_model_and_diffusion(args, data)
    if args.corr_noise :
        len_model.to(dist_util.dev())
        len_model.load_state_dict(torch.load(args.len_model_path)['model_state_dict'])
        len_model.eval()
    
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, len_model, diffusion, data, dataset).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
