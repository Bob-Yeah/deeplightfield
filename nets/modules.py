from typing import List, Tuple
import math
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity, Linear
from utils import device
from utils import sphere
from utils import misc
from utils.constants import *


class BatchLinear(nn.Linear):
    '''
    A linear meta-layer that can deal with batched weight matrices and biases,
    as for instance output by a hypernetwork.
    '''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        # if params is None:
        #    params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class FcLayer(nn.Module):

    def __init__(self, in_chns: int, out_chns: int, activation: str = 'linear', skip_chns: int = 0):
        super().__init__()
        nls_and_inits = {
            'sine': (Sine(), sine_init),
            'relu': (nn.ReLU(), None),
            'sigmoid': (nn.Sigmoid(), None),
            'tanh': (nn.Tanh(), None),
            'selu': (nn.SELU(), init_weights_selu),
            'softplus': (nn.Softplus(), init_weights_normal),
            'elu': (nn.ELU(), init_weights_elu),
            'softmax': (nn.Softmax(dim=-1), softmax_init),
            'logsoftmax': (nn.LogSoftmax(dim=-1), softmax_init),
            'linear': (None, None)
        }
        nl, nl_weight_init = nls_and_inits[activation]

        self.net = nn.Sequential(
            nn.Linear(in_chns + skip_chns, out_chns),
            nl
        ) if nl else nn.Linear(in_chns + skip_chns, out_chns)
        self.skip = skip_chns != 0

        if nl_weight_init is not None:
            nl_weight_init(self.net if isinstance(self.net, nn.Linear) else self.net[0])
        else:
            self.init_weights(activation)

    def forward(self, x: torch.Tensor, x0: torch.Tensor = None) -> torch.Tensor:
        return self.net(torch.cat([x0, x], dim=-1) if self.skip else x)

    def init_weights(self, activation):
        linear_net = self.net if isinstance(self.net, nn.Linear) else self.net[0]
        nn.init.xavier_normal_(linear_net.weight, gain=nn.init.calculate_gain(activation))
        nn.init.zeros_(linear_net.bias)


class FcNet(nn.Module):

    def __init__(self, *, in_chns: int, out_chns: int, nf: int, n_layers: int,
                 skips: List[int] = [], activation: str = 'relu'):
        """
        Initialize a full-connection net

        :kwarg in_chns: channels of input
        :kwarg out_chns: channels of output
        :kwarg nf: number of features in each hidden layer
        :kwarg n_layers: number of layers
        :kwarg skips: create skip connections from input to layers in this list
        """
        super().__init__()

        self.layers = [FcLayer(in_chns, nf, activation)] + [
            FcLayer(nf, nf, activation, skip_chns=in_chns if i in skips else 0)
            for i in range(n_layers - 1)
        ]
        if out_chns > 0:
            self.layers.append(FcLayer(nf, out_chns))
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer{i}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer in self.layers:
            x = layer(x, x0)
        return x


class InputEncoder(nn.Module):

    def Get(multires, input_dims):
        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
        }
        return InputEncoder(**embed_kwargs)

    def __init__(self, **kwargs):
        super().__init__()
        self.in_dim = kwargs['input_dims']
        self.num_freqs = kwargs['num_freqs']
        self.out_dim = self.in_dim * self.num_freqs * 2
        self.include_input = kwargs['include_input'] or self.num_freqs == 0
        if self.include_input:
            self.out_dim += self.in_dim
        if self.num_freqs > 0:
            self.freq_bands = 2. ** torch.linspace(0, kwargs['max_freq_log2'], self.num_freqs,
                                                   device=device.default())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encode the given input to R^D space

        :param input `Tensor(..., C)`: input
        :return `Tensor(..., D): encoded
        :rtype: torch.Tensor
        """
        if self.num_freqs > 0:
            input_ = input.unsqueeze(-2)  # to (..., 1, C)
            input_ = input_ * self.freq_bands[:, None]  # (..., Ne, C)
            output = torch.stack([input_.sin(), input_.cos()], dim=-2).flatten(-3)
            if self.include_input:
                output = torch.cat([input, output], dim=-1)
        else:
            output = input
        return output


class Sampler(nn.Module):

    def __init__(self, *, depth_range: Tuple[float, float], n_samples: int,
                 perturb_sample: bool, spherical: bool, lindisp: bool, inverse_r: bool):
        """
        Initialize a Sampler module

        :param depth_range: depth range for sampler
        :param n_samples: count to sample along ray
        :param perturb_sample: perturb the sample depths
        :param lindisp: If True, sample linearly in inverse depth rather than in depth
        """
        super().__init__()
        self.lindisp = lindisp
        if self.lindisp:
            depth_range = (1 / depth_range[0], 1 / depth_range[1])
        self.r = torch.linspace(depth_range[0], depth_range[1], n_samples, device=device.default())
        step = (depth_range[1] - depth_range[0]) / (n_samples - 1)
        self.perturb_sample = perturb_sample
        self.spherical = spherical
        self.inverse_r = inverse_r
        self.upper = torch.clamp_min(self.r + step / 2, 0)
        self.lower = torch.clamp_min(self.r - step / 2, 0)
        self.samples = n_samples

    def forward(self, rays_o, rays_d):
        """
        Sample points along rays. return Spherical or Cartesian coordinates, 
        specified by `self.shperical`

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return `Tensor(B, N, 3)`: sampled points
        :return `Tensor(B, N)`: corresponding depths along rays
        """
        if self.perturb_sample:
            # stratified samples in those intervals
            t_rand = torch.rand_like(self.r)
            r = self.lower + (self.upper - self.lower) * t_rand
        else:
            r = self.r
        if self.lindisp:
            r = torch.reciprocal(r)

        if self.spherical:
            pts, depths = sphere.ray_sphere_intersect(rays_o, rays_d, r)
            sphers = sphere.cartesian2spherical(pts, inverse_r=self.inverse_r)
            return sphers, pts, depths
        else:
            return rays_o[..., None, :] + rays_d[..., None, :] * r[..., None], r


class NewSampler(nn.Module):

    def __init__(self, *, depth_range: Tuple[float, float], n_samples: int,
                 perturb_sample: bool, spherical: bool, lindisp: bool, inverse_r: bool,
                 include_prev_samples=True):
        """
        Initialize a Sampler module

        :param depth_range: depth range for sampler
        :param n_samples: count to sample along ray
        :param perturb_sample: perturb the sample depths
        :param lindisp: If True, sample linearly in inverse depth rather than in depth
        """
        super().__init__()
        self.lindisp = lindisp
        self.perturb_sample = perturb_sample
        self.spherical = spherical
        self.s_range = (1 / depth_range[0], 1 / depth_range[1]) if self.lindisp else depth_range
        self.s_vals = torch.linspace(self.s_range[0], self.s_range[1], n_samples,
                                     device=device.default())
        mids = 0.5 * (self.s_vals[..., 1:] + self.s_vals[..., :-1])
        self.upper = torch.cat([mids, self.s_vals[..., -1:]], dim=-1)
        self.lower = torch.cat([self.s_vals[..., 0:1], mids], dim=-1)
        self.samples = n_samples
        self.include_prev_samples = include_prev_samples

    def forward(self, rays_o, rays_d, s_vals=None, weights=None):
        """
        Sample points along rays. return Spherical or Cartesian coordinates, 
        specified by `self.shperical`

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return `Tensor(B, N, 3)`: sampled points
        :return `Tensor(B, N)`: corresponding depths along rays
        """
        if s_vals is not None:
            mid = 0.5 * (s_vals[..., 1:] + s_vals[..., :-1])
            weights = weights.clone().detach()
            weights = weights[..., 1:-1]
            s = self.sample_pdf(mid, weights, self.samples - s_vals.size(-1))
            s = torch.sort(torch.cat([s, s_vals], dim=-1) if self.include_prev_samples else s,
                           descending=self.lindisp)[0]
        else:
            if self.perturb_sample:
                # stratified samples in those intervals
                t_rand = torch.rand_like(self.s_vals.expand(rays_o.size(0), -1))
                s = self.lower + (self.upper - self.lower) * t_rand
            else:
                s = self.s_vals.expand(rays_o.size(0), -1)
        z = torch.reciprocal(s) if self.lindisp else s
        if self.spherical:
            pts, depths = sphere.ray_sphere_intersect(rays_o, rays_d, z)
            sphers = sphere.cartesian2spherical(pts, inverse_r=self.lindisp)
            return sphers, pts, depths, s
        else:
            return rays_o[..., None, :] + rays_d[..., None, :] * z[..., None], z, s

    def sample_pdf(self, bins, weights, N_samples, det=False):
        '''
        :param bins: tensor of shape [..., M+1], M is the number of bins
        :param weights: tensor of shape [..., M]
        :param N_samples: number of samples along each ray
        :param det: if True, will perform deterministic sampling
        :return: [..., N_samples]
        '''
        # Get pdf
        weights = weights + TINY_FLOAT      # prevent nans
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
        cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
        cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

        # Take uniform samples
        dots_sh = list(weights.shape[:-1])
        M = weights.shape[-1]

        min_cdf = 0.00
        max_cdf = 1.00       # prevent outlier samples

        # u: [..., N_samples]
        if det:
            u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
            u = u.view([1] * len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples, ])
        else:
            sh = dots_sh + [N_samples]
            u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf

        # Invert CDF
        # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
        above_inds = torch.sum(u[..., None] >= cdf[..., None, :M], dim=-1).long()

        # random sample inside each bin
        below_inds = torch.clamp(above_inds - 1, min=0)
        inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

        cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])   # [..., N_samples, M+1]
        cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

        bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])    # [..., N_samples, M+1]
        bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

        # fix numeric issue
        denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
        denom = torch.where(denom < TINY_FLOAT, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom

        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_FLOAT)

        return samples


class PdfSampler(nn.Module):

    def __init__(self, *, depth_range: Tuple[float, float], n_samples: int,
                 perturb_sample: bool, spherical: bool, lindisp: bool, inverse_r: bool,
                 n_bins: int):
        """
        Initialize a Sampler module

        :param depth_range: depth range for sampler
        :param n_samples: count to sample along ray
        :param perturb_sample: perturb the sample depths
        :param lindisp: If True, sample linearly in inverse depth rather than in depth
        """
        super().__init__()
        self.lindisp = lindisp
        self.perturb_sample = perturb_sample
        self.spherical = spherical
        self.samples = n_samples
        self.n_bins = n_bins
        self.s_range = (1 / depth_range[0], 1 / depth_range[1]) if self.lindisp else depth_range
        self.s_vals = torch.linspace(self.s_range[0], self.s_range[1], n_bins,
                                     device=device.default())
        mids = 0.5 * (self.s_vals[..., 1:] + self.s_vals[..., :-1])
        self.bins = torch.cat([self.s_vals[..., 0:1], mids, self.s_vals[..., -1:]], dim=-1)
        self.upper = torch.cat([mids, self.s_vals[..., -1:]], dim=-1)
        self.lower = torch.cat([self.s_vals[..., 0:1], mids], dim=-1)

    def forward(self, rays_o, rays_d, weights):
        """
        Sample points along rays. return Spherical or Cartesian coordinates, 
        specified by `self.shperical`

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return `Tensor(B, N, 3)`: sampled points
        :return `Tensor(B, N)`: corresponding depths along rays
        """
        s = self.sample_pdf(self.bins, weights, self.samples)
        s = torch.sort(s, descending=self.lindisp)[0]
        z = torch.reciprocal(s) if self.lindisp else s
        if self.spherical:
            pts, depths = sphere.ray_sphere_intersect(rays_o, rays_d, z)
            sphers = sphere.cartesian2spherical(pts, inverse_r=self.lindisp)
            return sphers, pts, depths, s
        else:
            return rays_o[..., None, :] + rays_d[..., None, :] * z[..., None], z, s

    def sample_pdf(self, bins, weights, N_samples, det=True):
        '''
        :param bins: tensor of shape [..., M+1], M is the number of bins
        :param weights: tensor of shape [..., M]
        :param N_samples: number of samples along each ray
        :param det: if True, will perform deterministic sampling
        :return: [..., N_samples]
        '''
        # Get pdf
        weights = weights + TINY_FLOAT      # prevent nans
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
        cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
        cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

        # Take uniform samples
        dots_sh = list(weights.shape[:-1])
        M = weights.shape[-1]

        min_cdf = 0.00
        max_cdf = 1.00       # prevent outlier samples

        # u: [..., N_samples]
        if det:
            u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
            u = u.view([1] * len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples, ])
        else:
            sh = dots_sh + [N_samples]
            u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf

        # Invert CDF
        # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
        above_inds = torch.sum(u[..., None] >= cdf[..., None, :M], dim=-1).long()

        # random sample inside each bin
        below_inds = torch.clamp(above_inds - 1, min=0)
        inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

        cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])   # [..., N_samples, M+1]
        cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

        bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])    # [..., N_samples, M+1]
        bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

        # fix numeric issue
        denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
        denom = torch.where(denom < TINY_FLOAT, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom

        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_FLOAT)

        return samples


class AdaptiveSampler(nn.Module):

    def __init__(self, *, depth_range: Tuple[float, float], n_samples: int,
                 perturb_sample: bool, spherical: bool, lindisp: bool,
                 inverse_r: bool, n_bins: int, include_neighbor_bins=True):
        """
        Initialize a Sampler module

        :param depth_range: depth range for sampler
        :param n_samples: count to sample along ray
        :param perturb_sample: perturb the sample depths
        :param lindisp: If True, sample linearly in inverse depth rather than in depth
        """
        super().__init__()
        self.lindisp = lindisp
        self.perturb_sample = perturb_sample
        self.spherical = spherical
        self.samples = n_samples
        self.n_bins = n_bins
        self.include_neighbor_bins=include_neighbor_bins
        self.range = (1 / depth_range[0], 1 / depth_range[1]) if self.lindisp else depth_range
        self.bin_lower, self.bin_center, self.bin_upper = self._get_bounds(
            self.range, self.n_bins)
        self.s_lower, self.s_center, self.s_upper = self._get_bounds((0, 1), self.samples)

    def forward(self, rays_o, rays_d, depth, bins):
        """
        Sample points along rays. return Spherical or Cartesian coordinates, 
        specified by `self.shperical`

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param depth `Tensor(B)`: rays' ref depth
        :return `Tensor(B, N, 3)`: sampled points
        :return `Tensor(B, N)`: corresponding depths along rays
        """
        if self.perturb_sample:
            t = torch.rand_like(self.s_center.expand(rays_o.size(0), -1))  # (B, N)
        else:
            t = self.s_center  # (N)

        if depth is not None:
            r = sphere.cartesian2spherical(
                rays_o + rays_d * depth[:, None])[:, 0] if self.spherical else depth
            if self.lindisp:
                r = torch.reciprocal(r)
            bin = (r - self.range[0]) / (self.range[1] - self.range[0]) * (self.n_bins - 1)
        else:
            bin = (bins[..., 0] - 0.5) * 2 * (self.n_bins - 1)
        if self.include_neighbor_bins:
            bin_below = torch.clamp_min(bin - 1, 0).to(torch.long)
            bin_above = torch.clamp_max(bin + 1, self.n_bins - 1).to(torch.long)
        else:
            bin_below = bin_above = bin.to(torch.long)
        sample_lower = self.bin_lower[bin_below][:, None]  # (B, 1)
        sample_upper = self.bin_upper[bin_above][:, None]  # (B, 1)
        s = sample_lower + (sample_upper - sample_lower) * t  # (B, N)
        z = torch.reciprocal(s) if self.lindisp else s  # (B, N)
        if self.spherical:
            pts, depths = sphere.ray_sphere_intersect(rays_o, rays_d, z)
            sphers = sphere.cartesian2spherical(pts, inverse_r=self.lindisp)
            return sphers, pts, depths, s
        else:
            return rays_o[..., None, :] + rays_d[..., None, :] * z[..., None], z, s

    def _get_bounds(self, range, n):
        center = torch.linspace(range[0], range[1], n, device=device.default())
        mids = 0.5 * (center[..., 1:] + center[..., :-1])
        upper = torch.cat([mids, center[..., -1:]], dim=-1)
        lower = torch.cat([center[..., 0:1], mids], dim=-1)
        return lower, center, upper


class Rendering(nn.Module):

    def __init__(self, *, raw_noise_std: float = 0.0, white_bg: bool = False):
        """
        Initialize a Rendering module
        """
        super().__init__()
        self.raw_noise_std = raw_noise_std
        self.white_bg = white_bg

    def forward(self, raw, z_vals, ret_extra: bool = False):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        color, alpha = self.raw2color(raw, z_vals)

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        one_minus_alpha = misc.broadcast_cat(
            torch.cumprod(1 - alpha[..., :-1] + 1e-10, dim=-1),
            1.0, append=False)
        weights = alpha * one_minus_alpha  # (N_rays, N_samples)

        # (N_rays, 1|3), computed weighted color of each sample along each ray.
        color_map = torch.sum(weights[..., None] * color, dim=-2)

        # To composite onto a white background, use the accumulated alpha map.
        if self.white_bg or ret_extra:
            # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
            acc_map = torch.sum(weights, -1)
            if self.white_bg:
                color_map = color_map + (1. - acc_map[..., None])
        else:
            acc_map = None

        if not ret_extra:
            return color_map

        # Estimated depth map is expected distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        disp_map = torch.clamp_min(
            depth_map / torch.sum(weights, dim=-1), 1e-10).reciprocal()

        return color_map, disp_map, acc_map, weights, depth_map

    def raw2color(self, raw: torch.Tensor, z_vals: torch.Tensor):
        """
        Raw value inferred from model to color and alpha

        :param raw `Tensor(N.rays, N.samples, 2|4)`: model's output
        :param z_vals `Tensor(N.rays, N.samples)`: integration time
        :return `Tensor(N.rays, N.samples, 1|3)`: color
        :return `Tensor(N.rays, N.samples)`: alpha
        """

        def raw2alpha(raw, dists, act_fn=torch.relu):
            """
            Function for computing density from model prediction.
            This value is strictly between [0, 1].
            """
            return -torch.exp(-act_fn(raw) * dists) + 1.0

        # Compute 'distance' (in time) between each integration time along a ray.
        # The 'distance' from the last integration time is infinity.
        # dists: (N_rays, N_samples)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        last_dist = torch.zeros_like(z_vals[..., 0:1]) + TINY_FLOAT
        dists = torch.cat([dists, last_dist], -1)

        # Extract RGB of each sample position along each ray.
        color = torch.sigmoid(raw[..., :-1])  # (N_rays, N_samples, 1|3)

        if self.raw_noise_std > 0.:
            # Add noise to model's predictions for density. Can be used to
            # regularize network during training (prevents floater artifacts).
            noise = torch.normal(0.0, self.raw_noise_std, raw[..., -1].size())
            alpha = raw2alpha(raw[..., -1] + noise, dists)
        else:
            alpha = raw2alpha(raw[..., -1], dists)

        return color, alpha


class Mlp(nn.Module):

    def __init__(self, *, coord_chns, density_chns, color_chns, core_nf, core_layers,
                 dir_chns=0, dir_nf=0, activation='relu'):
        super().__init__()
        self.core = FcNet(in_chns=coord_chns, out_chns=0, nf=core_nf, n_layers=core_layers,
                          activation=activation)
        self.density_out = nn.Linear(core_nf, density_chns) if density_chns > 0 else None
        if color_chns == 0:
            self.color_out = None
        elif dir_chns > 0:
            self.color_out = FcNet(in_chns=core_nf + dir_chns, out_chns=color_chns, nf=dir_nf,
                                   n_layers=1, activation=activation)
        else:
            self.color_out = nn.Linear(core_nf, color_chns)
            self.color_out = nn.Sequential(self.color_out, nn.Sigmoid())

    def forward(self, coord: torch.Tensor, dir: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        core_output = self.core(coord)
        density = self.density_out(core_output) if self.density_out is not None else None
        if self.color_out is None:
            color = None
        else:
            if dir is not None:
                core_output = torch.cat([core_output, dir], dim=-1)
            color = self.color_out(core_output)
        return color, density


class NewMlp(nn.Module):

    def __init__(self, *, coord_chns, density_chns, color_chns, core_nf, core_layers,
                 dir_chns=0, dir_nf=0, activation='relu', skips=[]):
        super().__init__()
        self.core = FcNet(in_chns=coord_chns, out_chns=0, nf=core_nf, n_layers=core_layers,
                          skips=skips, activation=activation)
        self.density_out = FcLayer(core_nf, density_chns) if density_chns > 0 else None
        if color_chns == 0:
            self.feature_out = None
            self.color_out = None
        elif dir_chns > 0:
            self.feature_out = FcLayer(core_nf, core_nf)
            self.color_out = nn.Sequential(
                FcLayer(core_nf + dir_chns, dir_nf, activation),
                FcLayer(dir_nf, color_chns, 'sigmoid')
            )
        else:
            self.feature_out = Identity()
            self.color_out = FcLayer(core_nf, color_chns, 'sigmoid')

    def forward(self, coord: torch.Tensor, dir: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        core_output = self.core(coord)
        density = self.density_out(core_output) if self.density_out is not None else None
        if self.color_out is None:
            color = None
        else:
            feature_output = self.feature_out(core_output)
            if dir is not None:
                feature_output = torch.cat([feature_output, dir], dim=-1)
            color = self.color_out(feature_output)
        return color, density


class AlphaComposition(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, color, alpha, bg=None):
        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        one_minus_alpha = torch.cumprod(1 - alpha[..., :-1] + TINY_FLOAT, dim=-1)
        one_minus_alpha = torch.cat([
            torch.ones_like(one_minus_alpha[..., 0:1]),
            one_minus_alpha
        ], dim=-1)
        weights = alpha * one_minus_alpha  # (N_rays, N_samples)

        # (N_rays, 1|3), computed weighted color of each sample along each ray.
        color_map = torch.sum(weights[..., None] * color, dim=-2)

        # To composite onto a white background, use the accumulated alpha map.
        if bg is not None:
            # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
            acc_map = torch.sum(weights, -1)
            color_map = color_map + bg * (1. - acc_map[..., None])

        return {
            'color': color_map,
            'weight': weights,
        }


class NewRendering(nn.Module):

    def __init__(self, *, raw_noise_std=0.0):
        """
        Initialize a Rendering module
        """
        super().__init__()
        self.alpha_composition = AlphaComposition()
        self.raw_noise_std = raw_noise_std

    def forward(self, color, density, z_vals, bg_color=None, ret_depth=False, debug=False):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          color: [num_rays, num_samples along ray, 1|3]. Predicted color from model.
          density: [num_rays, num_samples along ray]. Predicted density from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.

        Returns:
          rgb_map: [num_rays, 1|3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        alpha = self.density2alpha(density, z_vals)
        color_map, weights = misc.values(self.alpha_composition(
            color, alpha, bg_color), 'color', 'weight')
        ret = {
            'color': color_map,
            'weight': weights
        }
        if ret_depth:
            ret['depth'] = torch.sum(weights * z_vals, dim=-1)
        if debug:
            ret['layers'] = torch.cat([color, alpha[..., None]], dim=-1)
        return ret

    def density2alpha(self, density: torch.Tensor, z_vals: torch.Tensor):
        """
        Raw value inferred from model to color and alpha

        :param density `Tensor(N.rays, N.samples)`: model's output density
        :param z_vals `Tensor(N.rays, N.samples)`: integration time
        :return `Tensor(N.rays, N.samples)`: alpha
        """

        # Compute 'distance' (in time) between each integration time along a ray.
        # The 'distance' from the last integration time is infinity.
        # dists: (N_rays, N_samples)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        last_dist = torch.zeros_like(z_vals[..., 0:1]) + TINY_FLOAT
        dists = torch.cat([dists, last_dist], -1)

        if self.raw_noise_std > 0.:
            # Add noise to model's predictions for density. Can be used to
            # regularize network during training (prevents floater artifacts).
            noise = torch.normal(0.0, self.raw_noise_std, density.size())
            density = density + noise
        return -torch.exp(-torch.relu(density) * dists) + 1.0


########################
# Initialization methods


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-math.sqrt(6 / num_input) / 30, math.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def softmax_init(m):
    with torch.no_grad():
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.constant_(m.bias,val=0)