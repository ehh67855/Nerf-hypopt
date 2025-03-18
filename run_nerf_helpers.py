import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
                
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        Neural Radiance Field (NeRF) Model
        """
        super(NeRF, self).__init__()
        self.D = D  # Number of layers
        self.W = W  # Neurons per layer
        self.input_ch = input_ch  # Input channels for position
        self.input_ch_views = input_ch_views  # Input channels for view direction
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # Point embedding layers
        self.pts_linears = nn.ModuleList()
        # First layer
        self.pts_linears.append(nn.Linear(self.input_ch, self.W))
        # Hidden layers
        for i in range(1, D):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(self.W + self.input_ch, self.W))
            else:
                self.pts_linears.append(nn.Linear(self.W, self.W))
        
        if use_viewdirs:
            # Feature linear layer
            self.feature_linear = nn.Linear(self.W, self.W)
            # Alpha linear layer
            self.alpha_linear = nn.Linear(self.W, 1)
            # View direction layers
            self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.W, self.W // 2)])
            # RGB linear layer
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            # Output linear layer
            self.output_linear = nn.Linear(self.W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)
            h = self.pts_linears[i](h)
            h = F.relu(h)
    
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
    
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
    
        return outputs

    def load_weights_from_keras(self, weights):
        raise NotImplementedError("This function is not implemented.")

# Ray helpers
def get_rays(H, W, K, c2w):
    """Get ray origins and directions for a pinhole camera."""
    device = K.device  # Ensure tensors are on the correct device
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device)
    )
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0],
                        -(j - K[1][2]) / K[1][1],
                        -torch.ones_like(i, device=device)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    """
    Get ray origins, directions in numpy.
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2])/K[0][0], 
                     -(j - K[1][2])/K[1][1], 
                     -np.ones_like(i)], -1)  # (H, W, 3)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # (H, W, 3)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))  # (H, W, 3)
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from camera space to NDC space.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Hierarchical sampling.
    """
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (batch, len(bins)-1)
    cdf = torch.cumsum(pdf, -1)  # (batch, len(bins)-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])  # (batch, N_samples)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)  # (batch, N_samples)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        u = np.random.rand(*list(cdf.shape[:-1]) + [N_samples])
        u = torch.Tensor(u).to(device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, u.shape[-1], -1), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, u.shape[-1], -1), 2, inds_g)

    denom = cdf_g[...,1] - cdf_g[...,0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom).to(device), denom)
    t = (u - cdf_g[...,0]) / denom
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])

    return samples
