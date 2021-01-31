from typing import List, Tuple
import torch
import torch.nn as nn
from ..my import device
from ..my import util


class FcLayer(nn.Module):

    def __init__(self, in_chns: int, out_chns: int, activate: nn.Module = None,
                 skip_chns: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_chns + skip_chns, out_chns),
            activate
        ) if activate else nn.Linear(in_chns + skip_chns, out_chns)
        self.skip = skip_chns != 0

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x0, x], dim=-1) if self.skip else x)


class FcNet(nn.Module):

    def __init__(self, *, in_chns: int, out_chns: int,
                 nf: int, n_layers: int, skips: List[int] = []):
        """
        Initialize a full-connection net

        :kwarg in_chns: channels of input
        :kwarg out_chns: channels of output
        :kwarg nf: number of features in each hidden layer
        :kwarg n_layers: number of layers
        :kwarg skips: create skip connections from input to layers in this list
        """
        super().__init__()
        self.layers = list()
        self.layers += [FcLayer(in_chns, nf, nn.ReLU())]
        self.layers += [
            FcLayer(nf, nf, nn.ReLU(),
                    skip_chns=in_chns if i in skips else 0)
            for i in range(1, n_layers)
        ]
        self.layers += [FcLayer(nf, out_chns)]
        for i, layer in enumerate(self.layers):
            self.add_module('layer%d' % i, layer)
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer in self.layers:
            x = layer(x, x0)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)


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
        self.include_input = kwargs['include_input']
        self.in_dim = kwargs['input_dims']
        self.out_dim = self.in_dim * kwargs['num_freqs'] * 2
        if self.include_input:
            self.out_dim += self.in_dim
        self.freq_bands = 2. ** torch.linspace(
            0, kwargs['max_freq_log2'], kwargs['num_freqs'],
            device=device.GetDevice())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encode the given input to R^D space

        :param input ```Tensor(..., C)```: input
        :return ```Tensor(..., D): encoded
        :rtype: torch.Tensor
        """
        input_ = input.unsqueeze(-2)  # to (..., 1, C)
        input_ = input_ * self.freq_bands[:, None]  # (..., Ne, C)
        output = torch.stack([input_.sin(), input_.cos()], dim=-2).flatten(-3)
        if self.include_input:
            output = torch.cat([input, output], dim=-1)
        return output


rand_gen = torch.Generator(device=device.GetDevice())
rand_gen.manual_seed(torch.seed())


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
        self.r = torch.linspace(depth_range[0], depth_range[1],
                                n_samples, device=device.GetDevice())
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
        specified by ```self.shperical```

        :param rays_o ```Tensor(B, 3)```: rays' origin
        :param rays_d ```Tensor(B, 3)```: rays' direction
        :return ```Tensor(B, N, 3)```: sampled points
        :return ```Tensor(B, N)```: corresponding depths along rays
        """
        if self.perturb_sample:
            # stratified samples in those intervals
            t_rand = torch.rand(self.r.size(),
                                generator=rand_gen,
                                device=device.GetDevice())
            r = self.lower + (self.upper - self.lower) * t_rand
        else:
            r = self.r
        if self.lindisp:
            r = torch.reciprocal(r)

        if self.spherical:
            pts, depths = util.RaySphereIntersect(rays_o, rays_d, r)
            sphers = util.CartesianToSpherical(pts, inverse_r=self.inverse_r)
            return sphers, pts, depths
        else:
            return rays_o[..., None, :] + rays_d[..., None, :] * r[..., None], r


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
        one_minus_alpha = util.broadcast_cat(
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

        :param raw ```Tensor(N.rays, N.samples, 2|4)```: model's output
        :param z_vals ```Tensor(N.rays, N.samples)```: integration time
        :return ```Tensor(N.rays, N.samples, 1|3)```: color
        :return ```Tensor(N.rays, N.samples)```: alpha
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
        last_dist = z_vals[..., 0:1] * 0 + 1e10

        dists = torch.cat([
            dists, last_dist
        ], -1)

        # Extract RGB of each sample position along each ray.
        color = torch.sigmoid(raw[..., :-1])  # (N_rays, N_samples, 1|3)

        if self.raw_noise_std > 0.:
            # Add noise to model's predictions for density. Can be used to
            # regularize network during training (prevents floater artifacts).
            noise = torch.normal(0.0, self.raw_noise_std,
                                 raw[..., 3].size(), rand_gen)
            alpha = raw2alpha(raw[..., -1] + noise, dists)
        else:
            alpha = raw2alpha(raw[..., -1], dists)

        return color, alpha
