from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
from LPM.model import LengthUNet
def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    if args.corr_noise :
        len_model = LengthUNet()
    else : 
        len_model = None
    return model, diffusion, len_model

def get_len_model_args(args):
    name                 = 'unet',
    dims                 = 1, # spatial dimension, if dims==1, [B x C x L], if dims==2, [B x C x W x H]
    length               = 30, 
    n_in_channels        = 128, # input channels
    n_base_channels      = 64, # base channel size
    n_emb_dim            = 128, # time embedding size
    n_cond_dim           = 0, # conditioning vector size (default is 0 indicating an unconditional model)
    n_time_dim           = 0,
    n_enc_blocks         = 3, # number of encoder blocks
    n_groups             = 16, # group norm paramter
    n_heads              = 4, # number of heads
    actv                 = nn.SiLU(),
    kernel_size          = 3, # kernel size
    padding              = 1,
    use_attention        = True,
    skip_connection      = True, # (optional) additional final skip connection
    use_scale_shift_norm = True, # positional embedding handling
    chnnel_multiples     = (1,2,4),
    updown_rates         = (2,2,2),
    device               = 'cpu',
    return 
def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 23
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'humanml2':
        # data_rep = 'hml_vec'
        njoints = 138
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif args.dataset == 'LAFAN':
        data_rep = 'rot6d'
        njoints = 138
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = args.pred_xstart  # we always predict x_start (a.k.a. x0), that's our deal!
    # predict_xstart = True
    # predict_xstart = False  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )
    
