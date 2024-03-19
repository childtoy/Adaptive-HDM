
import math,random
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn

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
    """ 
    U-Net for diffusion models (legacy)
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
        use_attention        = True,
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
        self.use_attention        = use_attention
        self.skip_connection      = skip_connection
        self.use_scale_shift_norm = use_scale_shift_norm
        self.chnnel_multiples     = chnnel_multiples
        self.updown_rates         = updown_rates
        self.device               = device
        
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
                    use_scale_shift_norm = self.use_scale_shift_norm,
                ).to(device)
            )
            # Attention block in encoder
            if self.use_attention:
                self.enc_layers.append(
                    AttentionBlock(
                        name           = 'att',
                        n_channels     = out_channel,
                        n_heads        = self.n_heads,
                        n_groups       = self.n_groups,
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
            nn.Linear(in_features=input_dim,out_features=1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.5), 
            nn.Linear(in_features=1024,out_features=32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.5), 
            nn.Linear(in_features=32,out_features=10)
        ).to(self.device)
        
        # Define U-net encoder
        self.enc_net = nn.Sequential()
        for l_idx,layer in enumerate(self.enc_layers):
            self.enc_net.add_module(
                name   = 'enc_%02d'%(l_idx),
                module = TimestepEmbedSequential(layer)
            )
        
    def forward(self,x,timesteps=None,c=None):
        """ 
        :param x: [B x n_in_channels x ...]
        :timesteps: [B]
        :return: [B x n_in_channels x ...], same shape as x
        """
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
        
        # Lift input
        hidden = self.lift(x) # [B x n_base_channels x ...]
        if isinstance(hidden,tuple): hidden = hidden[0] # in case of having tuple
        intermediate_output_dict['x_lifted'] = hidden
        
        # Encoder
        self.h_enc_list = [hidden] # start with lifted input
        for m_idx,module in enumerate(self.enc_net):
            hidden = module(hidden)
            if isinstance(hidden,tuple): hidden = hidden[0] # in case of having tuple
            # Append
            module_name = module[0].name
            intermediate_output_dict['h_enc_%s_%02d'%(module_name,m_idx)] = hidden
            # Append encoder output
            if self.use_attention:
                if (m_idx%2) == 1:
                    self.h_enc_list.append(hidden)
            else:
                self.h_enc_list.append(hidden)
            
        # Mid
        hidden = self.mid(hidden)
        if isinstance(hidden,tuple): hidden = hidden[0] # in case of having tuple
        
        # Projection
        hidden = th.flatten(hidden, start_dim=1)
        out = self.proj(hidden)
        
        return out