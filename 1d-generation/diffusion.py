import math,random
import numpy as np
import torch as th
import torch.nn as nn
from module import (
    conv_nd,
    ResBlock,
    AttentionBlock,
    TimestepEmbedSequential,
)

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

def forward_sample(x0_batch,t_batch,dc,M=None):
    """
    Forward diffusion sampling
    :param x0_batch: [B x C x ...]
    :param t_batch: [B]
    :param dc: dictionary of diffusion constants
    :param M: a matrix of [L x L] for [B x C x L] data
    :return: xt_batch of [B x C x ...] and noise of [B x C x ...]
    """
    # Gather diffusion constants with matching dimension
    out_shape = (t_batch.shape[0],) + ((1,)*(len(x0_batch.shape)-1))
    device = t_batch.device
    sqrt_alphas_bar_t = th.gather(
        input = th.from_numpy(dc['sqrt_alphas_bar']).to(device), # [T]
        dim   = -1,
        index = t_batch
    ).reshape(out_shape) # [B x 1 x 1 x 1] if (rank==4) and [B x 1 x 1] if (rank==3)
    sqrt_one_minus_alphas_bar = th.gather(
        input = th.from_numpy(dc['sqrt_one_minus_alphas_bar']).to(device), # [T]
        dim   = -1,
        index = t_batch
    ).reshape(out_shape) # [B x 1 x 1 x 1] if (rank==4) and [B x 1 x 1] if (rank==3)
    
    # Forward sample
    noise = th.randn_like(input=x0_batch) # [B x C x ...]
    
    # (optional) correlated noise
    if M is not None:
        B = x0_batch.shape[0]
        C = x0_batch.shape[1]
        L = x0_batch.shape[2]
        M_exp = 0
        
        if isinstance(M, list): # if M is a list,
            M_use = random.choice(M)
            if len(M_use.shape) == 3:
                M_exp = M_use[None] # [D x L x L] => [1 x D x L x L]
            else:
                M_exp = M_use[None,None,:,:].expand(B,C,L,L) # [L x L] => [B x C x L x L]                
        else:
            M_use = M # [L x L]
            M_exp = M_use[None,None,:,:].expand(B,C,L,L) # [L x L] => [B x C x L x L]                
            
        noise_exp = noise[:,:,:,None] # [B x C x L x 1]
        noise_exp = M_exp @ noise_exp # [B x C x L x 1]
        noise = noise_exp.squeeze(dim=3) # [B x C x L]

    # Jump diffusion
    xt_batch = sqrt_alphas_bar_t*x0_batch + \
        sqrt_one_minus_alphas_bar*noise # [B x C x ...]
    return xt_batch, noise

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class DiffusionUNetLegacy(nn.Module):
    """ 
    U-Net for diffusion models (legacy)
    """
    def __init__(
        self,
        name                 = 'unet',
        dims                 = 1, # spatial dimension, if dims==1, [B x C x L], if dims==2, [B x C x W x H]
        n_in_channels        = 128, # input channels
        n_base_channels      = 64, # base channel size
        n_emb_dim            = 128, # time embedding size
        n_cond_dim           = 0, # conditioning vector size (default is 0 indicating an unconditional model)
        n_enc_blocks         = 3, # number of encoder blocks
        n_dec_blocks         = 3, # number of decoder blocks
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
        self.n_enc_blocks         = n_enc_blocks
        self.n_dec_blocks         = n_dec_blocks
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
            
        # Decoder
        self.dec_layers = []
        for d_idx in range(self.n_dec_blocks):
            # Residual block in decoder
            if d_idx == 0: # first decoder
                in_channel = self.chnnel_multiples[::-1][d_idx]*self.n_base_channels + n_channels2cat.pop()
                out_channel = self.chnnel_multiples[::-1][d_idx]*self.n_base_channels
            else: 
                in_channel = self.chnnel_multiples[::-1][d_idx-1]*self.n_base_channels + n_channels2cat.pop()
                out_channel = self.chnnel_multiples[::-1][d_idx]*self.n_base_channels
                
            updown_rate = updown_rates[::-1][d_idx]
                
            self.dec_layers.append(
                ResBlock(
                    name                 = 'res',
                    n_channels           = in_channel,
                    n_emb_channels       = self.n_emb_dim,
                    n_out_channels       = out_channel,
                    n_groups             = self.n_groups,
                    dims                 = self.dims,
                    actv                 = self.actv,
                    kernel_size          = self.kernel_size,
                    padding              = self.padding,
                    upsample             = updown_rate != 1,
                    up_rate              = updown_rate,
                    use_scale_shift_norm = self.use_scale_shift_norm,
                ).to(device)
            )
            # Attention block in decoder
            if self.use_attention:
                self.dec_layers.append(
                    AttentionBlock(
                        name           = 'att',
                        n_channels     = out_channel,
                        n_heads        = self.n_heads,
                        n_groups       = self.n_groups,
                    ).to(device)
                )
            
        # Projection
        self.proj = conv_nd(
            dims         = self.dims,
            in_channels  = (1+self.chnnel_multiples[0])*self.n_base_channels,
            out_channels = self.n_in_channels,
            kernel_size  = 1,
        ).to(device)

        # Define U-net
        self.enc_net = nn.Sequential()
        for l_idx,layer in enumerate(self.enc_layers):
            self.enc_net.add_module(
                name   = 'enc_%02d'%(l_idx),
                module = TimestepEmbedSequential(layer)
            )
        self.dec_net = nn.Sequential()
        for l_idx,layer in enumerate(self.dec_layers):
            self.dec_net.add_module(
                name   = 'dec_%02d'%(l_idx),
                module = TimestepEmbedSequential(layer)
            )
        
    def forward(self,x,timesteps,c=None):
        """ 
        :param x: [B x n_in_channels x ...]
        :timesteps: [B]
        :return: [B x n_in_channels x ...], same shape as x
        """
        intermediate_output_dict = {}
        intermediate_output_dict['x'] = x
        
        # time embedding
        emb = self.time_embed(
            timestep_embedding(timesteps,self.n_base_channels)
        ) # [B x n_emb_dim]
        
        # conditional embedding
        if self.n_cond_dim > 0:
            cond = self.cond_embed(c)
            emb = emb + cond
        
        # Lift input
        h = self.lift(x) # [B x n_base_channels x ...]
        if isinstance(h,tuple): h = h[0] # in case of having tuple
        intermediate_output_dict['x_lifted'] = h
        
        # Encoder
        self.h_enc_list = [h] # start with lifted input
        for m_idx,module in enumerate(self.enc_net):
            h = module(h,emb)
            if isinstance(h,tuple): h = h[0] # in case of having tuple
            # Append
            module_name = module[0].name
            intermediate_output_dict['h_enc_%s_%02d'%(module_name,m_idx)] = h
            # Append encoder output
            if self.use_attention:
                if (m_idx%2) == 1:
                    self.h_enc_list.append(h)
            else:
                self.h_enc_list.append(h)
            
        # Mid
        h = self.mid(h)
        if isinstance(h,tuple): h = h[0] # in case of having tuple
        
        # Decoder
        for m_idx,module in enumerate(self.dec_net):
            if self.use_attention:
                if (m_idx%2) == 0:
                    h = th.cat((h,self.h_enc_list.pop()),dim=1)
            else:

                h = th.cat((h,self.h_enc_list.pop()),dim=1)
            h = module(h,emb)  # [B x n_base_channels x ...]
            if isinstance(h,tuple): h = h[0] # in cfase of having tuple
            # Append
            module_name = module[0].name
            intermediate_output_dict['h_dec_%s_%02d'%(module_name,m_idx)] = h
                
        # Projection
        h = th.cat((h,self.h_enc_list.pop()),dim=1)
        
        if self.skip_connection:
            out = self.proj(h) + x # [B x n_in_channels x ...]
        else:
            out = self.proj(h) # [B x n_in_channels x ...]
            
        # Append
        intermediate_output_dict['out'] = out # [B x n_in_channels x ...]
        
        return out,intermediate_output_dict
    
def eval_ddpm_1d(
    model,
    dc,
    n_sample,
    x_0,
    step_list_to_append,
    device,
    M           = None,
    hyp_len     = None,
    noise_scale = 1.0
    ):
    """
    Evaluate DDPM in 1D case
    :param model: score function
    :param dc: dictionary of diffusion coefficients
    :param n_sample: integer of how many trajectories to sample
    :param x_0: [N x C x L] tensor
    :param step_list_to_append: an ndarry of diffusion steps to append x_t
    """
    model.eval()
    n_data,C,L = x_0.shape
    
    # inbetweeing
    x_dummy = th.zeros(n_sample,C,L,device=device) # original code
    step_dummy = th.zeros(n_sample).type(th.long).to(device)
    _,x_T      = forward_sample(x_dummy,step_dummy,dc,M) # [n_sample x C x L]
    x_t        = x_T.clone() # [n_sample x C x L]
    x_t_list   = ['']*dc['T'] # empty list 

    for t in range(0,dc['T'])[::-1]: # 999 to 0
        # Score function
        step = th.full(
            size       = (n_sample,),
            fill_value = t,
            device     = device,
            dtype      = th.long) # [n_sample]
        with th.no_grad():
            eps_t,_ = model(x_t, step, hyp_len)
        betas_t = th.gather(
            input = th.from_numpy(dc['betas']).to(device), # [T]
            dim   = -1,
            index = step
        ).reshape((-1,1,1)) # [n_sample x 1 x 1]
        sqrt_one_minus_alphas_bar_t = th.gather(
            input = th.from_numpy(dc['sqrt_one_minus_alphas_bar']).to(device), # [T]
            dim   = -1,
            index = step
        ).reshape((-1,1,1)) # [n_sample x 1 x 1]
        sqrt_recip_alphas_t = th.gather(
            input = th.from_numpy(dc['sqrt_recip_alphas']).to(device), # [T]
            dim   = -1,
            index = step
        ).reshape((-1,1,1)) # [n_sample x 1 x 1]
        # Compute posterior mean
        mean_t = sqrt_recip_alphas_t * (
            x_t - betas_t*eps_t/sqrt_one_minus_alphas_bar_t
            ) # [n_sample x C x L]
        # Compute posterior variance
        posterior_variance_t = th.gather(
            input = th.from_numpy(dc['posterior_variance']).to(device), # [T]
            dim   = -1,
            index = step
        ).reshape((-1,1,1)) # [n_sample x 1 x 1]
        # Sample
        # add noise 
        # noise_scale = th.normal(0, 2, size=(1,))
        noise_scale = 1.0
        if t == 0: # last sampling, use mean
            x_t = mean_t
        else:
            _,noise_t = forward_sample(x_dummy,step_dummy,dc,M) # [n_sample x C x 1]
            x_t = mean_t + noise_scale*th.sqrt(posterior_variance_t)*noise_t
        # Append
        if t in step_list_to_append:
            x_t_list[t] = x_t
    model.train()
    return x_t_list # list of [n_sample x C x L]