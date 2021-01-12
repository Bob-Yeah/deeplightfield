from typing import Tuple
import torch
import torch.nn as nn
from .my import net_modules
from .my import util
from .my import device
from .my import color_mode


rand_gen = torch.Generator(device=device.GetDevice())
rand_gen.manual_seed(torch.seed())


def RaySphereIntersect(p: torch.Tensor, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Calculate intersections of each rays and each spheres

    :param p ```Tensor(B, 3)```: positions of rays
    :param v ```Tensor(B, 3)```: directions of rays
    :param r ```Tensor(N)```: , radius of spheres
    :return ```Tensor(B, N, 3)```: points of intersection
    :return ```Tensor(B, N)```: depths of intersection along ray
    """
    # p, v: Expand to (B, 1, 3)
    p = p.unsqueeze(1)
    v = v.unsqueeze(1)
    # pp, vv, pv: (B, 1)
    pp = (p * p).sum(dim=2)
    vv = (v * v).sum(dim=2)
    pv = (p * v).sum(dim=2)
    depths = (((pv * pv - vv * (pp - r * r)).sqrt() - pv) / vv)
    return p + depths[..., None] * v, depths


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
        if lindisp:
            self.r = 1 / torch.linspace(1 / depth_range[0], 1 / depth_range[1],
                                        n_samples, device=device.GetDevice())
        else:
            self.r = torch.linspace(depth_range[0], depth_range[1],
                                    n_samples, device=device.GetDevice())
        self.perturb_sample = perturb_sample
        self.spherical = spherical
        self.inverse_r = inverse_r
        if perturb_sample:
            mids = .5 * (self.r[1:] + self.r[:-1])
            self.upper = torch.cat([mids, self.r[-1:]], -1)
            self.lower = torch.cat([self.r[:1], mids], -1)

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

        if self.spherical:
            pts, depths = RaySphereIntersect(rays_o, rays_d, r)
            sphers = util.CartesianToSpherical(pts, inverse_r=self.inverse_r)
            return sphers, depths
        else:
            return rays_o[..., None, :] + rays_d[..., None, :] * r[..., None], r


class MslNet(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 color: int = color_mode.RGB,
                 encode_to_dim: int = 0,
                 export_mode: bool = False):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param gray: is grayscale mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.in_chns = 3
        self.input_encoder = net_modules.InputEncoder.Get(
            encode_to_dim, self.in_chns)
        fc_params['in_chns'] = self.input_encoder.out_dim
        fc_params['out_chns'] = 2 if color == color_mode.GRAY else 4
        self.sampler = Sampler(**sampler_params)
        self.rendering = Rendering()
        self.export_mode = export_mode
        if color == color_mode.YCbCr:
            self.net1 = net_modules.FcNet(
                in_chns=fc_params['in_chns'],
                out_chns=fc_params['nf'] + 2,
                nf=fc_params['nf'],
                n_layers=fc_params['n_layers'] - 2)
            self.net2 = net_modules.FcNet(
                in_chns=fc_params['nf'],
                out_chns=2,
                nf=fc_params['nf'],
                n_layers=1)
            self.net = None
        else:
            self.net = net_modules.FcNet(**fc_params)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                ret_depth: bool = False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o ```Tensor(B, 3)```: rays' origin
        :param rays_d ```Tensor(B, 3)```: rays' direction
        :return: ```Tensor(B, C)``, inferred images/pixels
        """
        coords, depths = self.sampler(rays_o, rays_d)
        encoded = self.input_encoder(coords)

        if not self.net:
            mid_output = self.net1(encoded)
            net2_output = self.net2(mid_output[..., :-2])
            raw = torch.cat([
                mid_output[..., -2:],
                net2_output
            ], -1)
        else:
            raw = self.net(encoded)
        
        if self.export_mode:
            colors, alphas = self.rendering.raw2color(raw, depths)
            return torch.cat([colors, alphas[..., None]], -1)

        if ret_depth:
            color_map, _, _, _, depth_map = self.rendering(
                raw, depths, ret_extra=True)
            return color_map, depth_map
        
        return self.rendering(raw, depths)
