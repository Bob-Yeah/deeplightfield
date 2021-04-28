import math
import torch
import torch.nn as nn
from .modules import *
from .bg_net import BgNet
from utils import sphere
from utils import color


class Nerf(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 normalize_coord: bool,
                 c: int = color.RGB,
                 coord_encode: int = 0,
                 dir_encode: int = None,
                 spherical_dir: bool = False,
                 bg_layer: bool = False):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param normalize_coord: whether normalize the spherical coords to [0, 2pi] before encode
        :param c: color mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.color = c
        self.normalize_coord = normalize_coord
        self.spherical_dir = spherical_dir
        self.n_samples = sampler_params['n_samples']
        self.coord_chns = 3
        self.color_chns = color.chns(self.color)

        self.coord_encoder = InputEncoder.Get(coord_encode, self.coord_chns)

        if dir_encode is not None:
            self.dir_chns = 2 if self.spherical_dir else 3
            self.dir_encoder = InputEncoder.Get(dir_encode, self.dir_chns)
        else:
            self.dir_chns = 0
            self.dir_encoder = None
        self.mlp = NewMlp(coord_chns=self.coord_encoder.out_dim,
                          density_chns=1,
                          color_chns=self.color_chns,
                          core_nf=fc_params['nf'],
                          core_layers=fc_params['n_layers'],
                          dir_chns=self.dir_encoder.out_dim if self.dir_encoder else 0,
                          dir_nf=fc_params['nf'] // 2,
                          activation=fc_params['activation'],
                          skips=fc_params['skips'])
        self.bg_net = BgNet({
            'nf': fc_params['nf'] // 2,
            'n_layers': fc_params['n_layers'],
            'activation': fc_params['activation']
        }, coord_encode, c) if bg_layer else None
        self.sampler = NewSampler(**sampler_params, include_prev_samples=True)
        self.rendering = NewRendering()
        if self.normalize_coord:
            self.register_buffer('angle_range', torch.tensor(
                [[1e5, 1e5], [-1e5, -1e5]]))
            self.register_buffer('depth_range', torch.tensor([
                self.sampler.lower[0], self.sampler.upper[-1]
            ]))

    def update_normalize_range(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        coords, _, _ = self.sampler(rays_o, rays_d)
        coords = coords[..., 1:].view(-1, 2)
        self.angle_range = torch.stack([
            torch.cat([coords, self.angle_range[0:1]]).amin(0),
            torch.cat([coords, self.angle_range[1:2]]).amax(0)
        ])

    def calc_local_dir(self, rays_d, coords, pts: torch.Tensor):
        """
        [summary]

        :param rays_d `Tensor(B, 3)`:
        :param coords `Tensor(B, N, 3)`:
        :param pts `Tensor(B, N, 3)`:
        :return `Tensor(B, N, 2)`
        """
        local_z = pts / pts.norm(dim=-1, keepdim=True)
        local_x = sphere.spherical2cartesian(
            coords + torch.tensor([0, math.radians(0.1), 0], device=coords.device)) - pts
        local_x = local_x / local_x.norm(dim=-1, keepdim=True)
        local_y = torch.cross(local_x, local_z, -1)
        local_rot = torch.stack(
            [local_x, local_y, local_z], dim=-2)  # (B, N, 3, 3)
        return sphere.cartesian2spherical(torch.matmul(rays_d[:, None, None, :], local_rot)) \
            .squeeze(-2)[..., 1:3]

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                prev_ret=None, ret_depth=False, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param prev_ret `Mapping`:
        :param ret_depth `bool`:
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        coords, pts, depths, s_vals = self.sampler(rays_o, rays_d) if prev_ret is None else \
            self.sampler(rays_o, rays_d, prev_ret['s'], prev_ret['weight'])
        coords_encoded = self.coord_encoder(coords)
        dirs_encoded = self.dir_encoder(rays_d)[:, None].expand(-1, self.n_samples, -1) \
            if self.dir_encoder is not None else None
        colors, densities = self.mlp(coords_encoded, dirs_encoded)
        bg_color = self.bg_net(rays_o, rays_d)['color'] if self.bg_net is not None else None

        # if self.normalize_coord:  # Normalize coords to [0, 2pi]
        #    range = torch.cat(
        #        [self.depth_range.view(2, 1), self.angle_range], 1)
        #    coords = (coords - range[0]) / (range[1] - range[0]) * 2 * math.pi
        ret = self.rendering(colors, densities[..., 0], depths, bg_color=bg_color,
                             ret_depth=ret_depth, debug=debug)
        ret['s'] = s_vals
        return ret


class CascadeNerf(nn.Module):

    def __init__(self, fc_params, sampler_params, fine_params,
                 normalize_coord: bool,
                 c: int = color.RGB,
                 coord_encode: int = 0,
                 dir_encode: int = None,
                 spherical_dir: bool = False,
                 bg_layer: bool = False):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param normalize_coord: whether normalize the spherical coords to [0, 2pi] before encode
        :param c: color mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.net1 = Nerf(fc_params, sampler_params, normalize_coord, c, coord_encode,
                         dir_encode, spherical_dir, bg_layer)
        fine_fc_params = fc_params.copy()
        fine_fc_params.update({
            'nf': fine_params['nf'],
            'n_layers': fine_params['n_layers']
        })
        fine_sampler_params = sampler_params.copy()
        fine_sampler_params.update({
            'n_samples': sampler_params['n_samples'] + fine_params['additional_samples']
        })
        self.net2 = Nerf(fine_fc_params, fine_sampler_params, normalize_coord, c, coord_encode,
                         dir_encode, spherical_dir, bg_layer) if fine_params['enable'] else None

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                ret_depth=False, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param s_vals `Tensor(B, N)`:
        :param weights `Tensor(B, N)`:
        :param ret_depth `bool`:
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        ret1 = self.net1(rays_o, rays_d, ret_depth=ret_depth, debug=debug)
        if self.net2 is None:
            return ret1
        ret2 = self.net2(rays_o, rays_d, prev_ret=ret1,
                         ret_depth=ret_depth, debug=debug)
        return ret1, ret2


class CascadeNerf2(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 normalize_coord: bool,
                 c: int = color.RGB,
                 coord_encode: int = 0,
                 dir_encode: int = None,
                 spherical_dir: bool = False,
                 bg_layer: bool = False):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param normalize_coord: whether normalize the spherical coords to [0, 2pi] before encode
        :param c: color mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.net = Nerf(fc_params, sampler_params, normalize_coord, c, coord_encode,
                        dir_encode, spherical_dir, bg_layer)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                ret_depth=False, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param s_vals `Tensor(B, N)`:
        :param weights `Tensor(B, N)`:
        :param ret_depth `bool`:
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        prev_ret = self.prev_net(rays_o, rays_d)
        return self.net(rays_o, rays_d, prev_ret=prev_ret, ret_depth=ret_depth, debug=debug)
