
import math,random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(n_channels,n_groups=1):
    """
    Make a standard normalization layer.

    :param n_channels: number of input channels.
    :param n_groups: number of groups. if this is 1, then it is identical to layernorm.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_groups=n_groups,num_channels=n_channels)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param n_channels: number of channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, 
        n_channels, 
        up_rate        = 2, # upsample rate
        up_mode        = 'nearest', # upsample mode ('nearest' or 'bilinear')
        use_conv       = False, # (optional) use output conv
        dims           = 2, # (optional) spatial dimension
        n_out_channels = None, # (optional) in case output channels are different from the input
        padding_mode   = 'zeros', 
        padding        = 1
    ):
        super().__init__()
        self.n_channels     = n_channels
        self.up_rate        = up_rate
        self.up_mode        = up_mode
        self.use_conv       = use_conv
        self.dims           = dims
        self.n_out_channels = n_out_channels or n_channels
        self.padding_mode   = padding_mode;
        self.padding        = padding
        
        if use_conv:
            self.conv = conv_nd(
                dims         = dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = 3, 
                padding      = padding,
                padding_mode = padding_mode)

    def forward(self, x):
        """ 
        :param x: [B x C x W x H]
        :return: [B x C x 2W x 2H]
        """
        assert x.shape[1] == self.n_channels
        if self.dims == 3: # 3D convolution
            x = F.interpolate(
                input = x,
                size  = (x.shape[2],x.shape[3]*2,x.shape[4]*2),
                mode  = self.up_mode
            )
        else:
            x = F.interpolate(
                input        = x,
                scale_factor = self.up_rate,
                mode         = self.up_mode
            ) # 'nearest' or 'bilinear'
            
        # (optional) final convolution
        if self.use_conv: 
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, 
        n_channels, 
        down_rate      = 2, # down rate
        use_conv       = False, # (optional) use output conv
        dims           = 2, # (optional) spatial dimension
        n_out_channels = None, # (optional) in case output channels are different from the input
        padding_mode   = 'zeros', 
        padding        = 1
    ):
        super().__init__()
        self.n_channels     = n_channels
        self.down_rate      = down_rate
        self.n_out_channels = n_out_channels or n_channels
        self.use_conv       = use_conv
        self.dims           = dims
        stride = self.down_rate if dims != 3 else (1, self.down_rate, self.down_rate)
        if use_conv:
            self.op = conv_nd(
                dims         = dims, 
                in_channels  = self.n_channels, 
                out_channels = self.n_out_channels,
                kernel_size  = 3, 
                stride       = stride,
                padding      = padding,
                padding_mode = padding_mode
            )
        else:
            assert self.n_channels == self.n_out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.n_channels
        return self.op(x)    

class ResBlock(nn.Sequential):
    """ 
    A residual block that can optionally change the number of channels and resolution
    
    :param n_channels: the number of input channels
    :param n_emb_channels: the number of timestep embedding channels
    :param n_out_channels: (if specified) the number of output channels
    :param n_groups: the number of groups in group normalization layer
    :param dims: spatial dimension
    :param p_dropout: the rate of dropout
    :param actv: activation
    :param use_conv: if True, and n_out_channels is specified, 
        use 3x3 conv instead of 1x1 conv
    :param use_scale_shift_norm: if True, use scale_shift_norm for handling emb
    :param upsample: if True, upsample
    :param downsample: if True, downsample
    :param sample_mode: upsample, downsample mode ('nearest' or 'bilinear')
    :param padding_mode: str
    :param padding: int
    """
    def __init__(
        self,
        name                 = 'resblock',
        n_channels           = 128,
        n_emb_channels       = 128,
        n_out_channels       = None,
        n_groups             = 16,
        dims                 = 2,
        p_dropout            = 0.5,
        kernel_size          = 3,
        actv                 = nn.SiLU(),
        use_conv             = False,
        upsample             = False,
        downsample           = False,
        up_rate              = 2,
        down_rate            = 2,
        sample_mode          = 'nearest',
        padding_mode         = 'zeros',
        padding              = 1,
    ):
        super().__init__()
        self.name                 = name
        self.n_channels           = n_channels
        self.n_emb_channels       = n_emb_channels
        self.n_groups             = n_groups
        self.dims                 = dims
        self.n_out_channels       = n_out_channels or self.n_channels
        self.kernel_size          = kernel_size
        self.p_dropout            = p_dropout
        self.actv                 = actv
        self.use_conv             = use_conv
        self.upsample             = upsample
        self.downsample           = downsample
        self.up_rate              = up_rate
        self.down_rate            = down_rate
        self.sample_mode          = sample_mode
        self.padding_mode         = padding_mode
        self.padding              = padding
        
        # Input layers
        self.in_layers = nn.Sequential(
            normalization(n_channels=self.n_channels,n_groups=self.n_groups),
            self.actv,
            conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = self.kernel_size,
                padding      = self.padding,
                padding_mode = self.padding_mode
            )
        )
        
        # Upsample or downsample
        self.updown = self.upsample or self.downsample
        if self.upsample:
            self.h_upd = Upsample(
                n_channels = self.n_channels,
                up_rate    = self.up_rate,
                up_mode    = self.sample_mode,
                dims       = self.dims)
            self.x_upd = Upsample(
                n_channels = self.n_channels,
                up_rate    = self.up_rate,
                up_mode    = self.sample_mode,
                dims       = self.dims)
        elif self.downsample:
            self.h_upd = Downsample(
                n_channels = self.n_channels,
                down_rate  = self.down_rate,
                dims       = self.dims)
            self.x_upd = Downsample(
                n_channels = self.n_channels,
                down_rate  = self.down_rate,
                dims       = self.dims)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()
            
        # Embedding layers
        self.emb_layers = nn.Sequential(
            self.actv,
            nn.Linear(
                in_features  = self.n_emb_channels,
                out_features = self.n_out_channels,
            ),
        )
        
        # Output layers
        self.out_layers = nn.Sequential(
            normalization(n_channels=self.n_out_channels,n_groups=self.n_groups),
            self.actv,
            nn.Dropout(p=self.p_dropout),
            zero_module(
                conv_nd(
                    dims         = self.dims, 
                    in_channels  = self.n_out_channels,
                    out_channels = self.n_out_channels,
                    kernel_size  = self.kernel_size,
                    padding      = self.padding,
                    padding_mode = self.padding_mode
                )
            ),
        )
        if self.n_channels == self.n_out_channels:
            self.skip_connection = nn.Identity()
            self.skip_connection_cond = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = self.kernel_size,
                padding      = self.padding,
                padding_mode = self.padding_mode
            )
            self.skip_connection_cond = conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = self.kernel_size,
                padding      = self.padding,
                padding_mode = self.padding_mode
            )
        else:
            self.skip_connection = conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = 1
            )
            self.skip_connection_cond = conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = 1
            )
        
    def forward(self,x):
        """
        :param x: [B x C x ...]
        :param emb: [B x n_emb_channels]
        :return: [B x C x ...]
        """
        # Input layer (groupnorm -> actv -> conv)
        if self.updown: # upsample or downsample
            in_norm_actv = self.in_layers[:-1]
            in_conv = self.in_layers[-1]
            h = in_norm_actv(x) 
            h = self.h_upd(h)
            h = in_conv(h)
            x = self.x_upd(x)
        else:
            h = self.in_layers(x) # [B x C x ...]
                        
        h = self.out_layers(h) # layernorm -> activation -> dropout -> conv
            
        # Skip connection
        out = h + self.skip_connection(x) # [B x C x ...]
        return out # [B x C x ...]
    


def get_named_beta_schedule(
    schedule_name,
    num_diffusion_timesteps, 
    scale_betas=1.0,
    np_type=np.float64
    ):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np_type
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_ddpm_constants(
    schedule_name = 'cosine',
    T             = 1000,
    np_type       = np.float64
    ):
    timesteps = np.linspace(start=1,stop=T,num=T)
    betas = get_named_beta_schedule(
        schedule_name           = schedule_name,
        num_diffusion_timesteps = T,
        scale_betas             = 1.0,
    ).astype(np_type) # [1,000]
    alphas                    = 1.0 - betas 
    alphas_bar                = np.cumprod(alphas, axis=0) #  cummulative product
    alphas_bar_prev           = np.append(1.0,alphas_bar[:-1])
    sqrt_recip_alphas         = np.sqrt(1.0/alphas)
    sqrt_alphas_bar           = np.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = np.sqrt(1.0-alphas_bar)
    posterior_variance        = betas*(1.0-alphas_bar_prev)/(1.0-alphas_bar)
    posterior_variance        = posterior_variance.astype(np_type)
    
    # Append
    dc = {}
    dc['schedule_name']             = schedule_name
    dc['T']                         = T
    dc['timesteps']                 = timesteps
    dc['betas']                     = betas
    dc['alphas']                    = alphas
    dc['alphas_bar']                = alphas_bar
    dc['alphas_bar_prev']           = alphas_bar_prev
    dc['sqrt_recip_alphas']         = sqrt_recip_alphas
    dc['sqrt_alphas_bar']           = sqrt_alphas_bar
    dc['sqrt_one_minus_alphas_bar'] = sqrt_one_minus_alphas_bar
    dc['posterior_variance']        = posterior_variance
    
    return dc


class lengthMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(lengthMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*2)
        self.bn3 = nn.BatchNorm1d(hidden_size*2)
        self.fc4 = nn.Linear(hidden_size*2, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = th.relu(self.bn1(self.fc1(x)))
        x = th.relu(self.bn2(self.fc2(x)))
        x = th.relu(self.bn3(self.fc3(x)))
        x = th.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

class LengthUNet(nn.Module):
=======
from module import (
    conv_nd,
    ResBlock,
    AttentionBlock,
    TimestepEmbedSequential,
)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionUNetLegacy(nn.Module):
>>>>>>> e20e5aa9570b99e1080fe4a168bb083d5df3ec04
    """ 
    U-Net for Length Prediction Modeuls
    """
    def __init__(
        self,
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
<<<<<<< HEAD
=======
        use_attention        = True,
>>>>>>> e20e5aa9570b99e1080fe4a168bb083d5df3ec04
        skip_connection      = True, # (optional) additional final skip connection
        use_scale_shift_norm = True, # positional embedding handling
        chnnel_multiples     = (1,2,4),
        updown_rates         = (2,2,2),
        device               = 'cpu',
    ):
        super().__init__()
        self.name                 = name
        self.dims                 = dims
        self.n_in_channels        = n_in_channels
        self.n_base_channels      = n_base_channels
        self.n_emb_dim            = n_emb_dim
        self.n_cond_dim           = n_cond_dim
        self.n_time_dim           = n_time_dim
        self.n_enc_blocks         = n_enc_blocks
        self.n_groups             = n_groups
        self.n_heads              = n_heads
        self.actv                 = actv
        self.kernel_size          = kernel_size
        self.padding              = padding
        self.skip_connection      = skip_connection
        self.use_scale_shift_norm = use_scale_shift_norm
        self.chnnel_multiples     = chnnel_multiples
        self.updown_rates         = updown_rates
        self.device               = device
<<<<<<< HEAD
                    
=======
        
        # Time embedding
        if self.n_time_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(in_features=self.n_base_channels,out_features=self.n_emb_dim),
                nn.SiLU(),
                nn.Linear(in_features=self.n_emb_dim,out_features=self.n_emb_dim),
            ).to(self.device)
            
        # Conditional embedding
        if self.n_cond_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(in_features=self.n_cond_dim,out_features=self.n_emb_dim),
                nn.SiLU(),
                nn.Linear(in_features=self.n_emb_dim,out_features=self.n_emb_dim),
            ).to(self.device)
            
>>>>>>> e20e5aa9570b99e1080fe4a168bb083d5df3ec04
        # Lifting (1x1 conv)
        self.lift = conv_nd(
            dims         = self.dims,
            in_channels  = self.n_in_channels,
            out_channels = self.n_base_channels,
            kernel_size  = 1,
        ).to(device)
        
        # Encoder
        self.enc_layers = []
        n_channels2cat = [] # channel size to concat for decoder (note that we should use .pop() )
        for e_idx in range(self.n_enc_blocks): # for each encoder block
            if e_idx == 0:
                in_channel  = self.n_base_channels
                out_channel = self.n_base_channels*self.chnnel_multiples[e_idx]
            else:
                in_channel  = self.n_base_channels*self.chnnel_multiples[e_idx-1]
                out_channel = self.n_base_channels*self.chnnel_multiples[e_idx]
            n_channels2cat.append(out_channel) # append out_channel
            updown_rate = updown_rates[e_idx]
            
            # Residual block in encoder
            self.enc_layers.append(
                ResBlock(
                    name                 = 'res',
                    n_channels           = in_channel,
                    n_emb_channels       = self.n_emb_dim,
                    n_out_channels       = out_channel,
                    n_groups             = self.n_groups,
                    dims                 = self.dims,
                    p_dropout            = 0.7,
                    actv                 = self.actv,
                    kernel_size          = self.kernel_size,
                    padding              = self.padding,
                    downsample           = updown_rate != 1,
                    down_rate            = updown_rate,
                ).to(device)
            )
        
        # Mid
        self.mid = conv_nd(
            dims         = self.dims,
            in_channels  = self.n_base_channels*self.chnnel_multiples[-1],
            out_channels = self.n_base_channels*self.chnnel_multiples[-1],
            kernel_size  = 1,
        ).to(device)
        
        if length == 32:
            input_dim = 4096
        else:
            input_dim = 7168 # when frames 60: latent dim (B, 1024, 7) -> flatten (1024*7) = (7168)
            
        # For Classification
        self.proj = nn.Sequential(
            nn.Dropout(0.25),
<<<<<<< HEAD
            nn.Linear(in_features=self.n_base_channels*self.chnnel_multiples[-1],
                      out_features=self.n_base_channels*self.chnnel_multiples[-1]//4),
=======
            nn.Linear(in_features=input_dim,out_features=1024),
>>>>>>> e20e5aa9570b99e1080fe4a168bb083d5df3ec04
            nn.GELU(),
            nn.LayerNorm(self.n_base_channels*self.chnnel_multiples[-1]//4),
            nn.Dropout(0.25), 
            nn.Linear(in_features=self.n_base_channels*self.chnnel_multiples[-1]//4
                      ,out_features=self.n_base_channels*self.chnnel_multiples[-1]//16),
            nn.GELU(),
            nn.LayerNorm(self.n_base_channels*self.chnnel_multiples[-1]//16),
            nn.Dropout(0.25), 
            nn.Linear(in_features=self.n_base_channels*self.chnnel_multiples[-1]//16,out_features=10)
        ).to(self.device)
        
        # Define U-net encoder
        self.enc_net = nn.Sequential()
        for l_idx,layer in enumerate(self.enc_layers):
            self.enc_net.add_module(
                name   = 'enc_%02d'%(l_idx),
                module = layer
            )
        
    def forward(self,x):
        """ 
        :param x: [B x n_in_channels x ...]
        :timesteps: [B]
        :return: [B x n_in_channels x ...], same shape as x
        """
<<<<<<< HEAD
        # intermediate_output_dict = {}
        # intermediate_output_dict['x'] = x
                
=======
        intermediate_output_dict = {}
        intermediate_output_dict['x'] = x
        
        emb = th.Tensor(0).to(self.device) 
        
        if self.n_time_dim > 0:
        # time embedding        
            emb = self.time_embed(
                timestep_embedding(timesteps,self.n_base_channels)
            ) # [B x n_emb_dim]
            
        # conditional embedding
        if self.n_cond_dim > 0:
            cond = self.cond_embed(c)
            emb = emb + cond
        
>>>>>>> e20e5aa9570b99e1080fe4a168bb083d5df3ec04
        # Lift input
        hidden = self.lift(x) # [B x n_base_channels x ...]
        if isinstance(hidden,tuple): hidden = hidden[0] # in case of having tuple
        # intermediate_output_dict['x_lifted'] = hidden
        
        # Encoder
        self.h_enc_list = [hidden] # start with lifted input
        for m_idx,module in enumerate(self.enc_net):
            hidden = module(hidden)
            if isinstance(hidden,tuple): hidden = hidden[0] # in case of having tuple
            # Append
            # module_name = module[0].name
            # intermediate_output_dict['h_enc_%s_%02d'%(module_name,m_idx)] = hidden
            self.h_enc_list.append(hidden)
            
        # Mid
        print(hidden.shape)
        hidden = self.mid(hidden)
        print(hidden.shape)
        if isinstance(hidden,tuple): hidden = hidden[0] # in case of having tuple
        
        # Projection
        hidden = th.flatten(hidden, start_dim=1)
        print(hidden.shape)
        out = self.proj(hidden)
        
        return out