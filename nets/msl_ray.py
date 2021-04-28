import math
import torch
import torch.nn as nn
from .modules import *
from utils import sphere
from utils import color


class MslRay(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 normalize_coord: bool,
                 c: int = color.RGB,
                 encode_to_dim: int = 0,
                 export_mode: bool = False):
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
        self.n_samples = sampler_params['n_samples']
        self.export_mode = export_mode
        self.normalize_coord = normalize_coord

        #self.input_encoder = InputEncoder.Get(encode_to_dim, 6 + self.n_samples) # v1
        self.input_encoder = InputEncoder.Get(encode_to_dim, self.n_samples * 3) # v2
        self.sampler = Sampler(**sampler_params)
        self.mlp = Mlp(coord_chns=self.input_encoder.out_dim,
                       density_chns=self.n_samples,
                       color_chns=self.n_samples * color.chns(self.color),
                       core_nf=fc_params['nf'],
                       core_layers=fc_params['n_layers'])
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
        local_rot = torch.stack([local_x, local_y, local_z], dim=-2)  # (B, N, 3, 3)
        return sphere.cartesian2spherical(torch.matmul(rays_d[:, None, None, :], local_rot)) \
            .squeeze(-2)[..., 1:3]

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, ret_depth: bool = False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        _, pts, depths = self.sampler(rays_o, rays_d)
        #coords = torch.cat([pts[:, 0], rays_d, depths], 1) # v1
        coords = pts.flatten(1, 2) # v2
        # if self.normalize_coord:  # Normalize coords to [0, 2pi]
        #    range = torch.cat(
        #        [self.depth_range.view(2, 1), self.angle_range], 1)
        #    coords = (coords - range[0]) / (range[1] - range[0]) * 2 * math.pi
        encoded = self.input_encoder(coords)
        colors, densities = self.mlp(encoded)
        rendering_ret = self.rendering(colors.view(-1, self.n_samples, color.chns(self.color)),
                                       densities, depths, ret_extra=ret_depth)
        if ret_depth:
            return rendering_ret[0], rendering_ret[-1]
        return rendering_ret


class ExportNet(nn.Module):

    def __init__(self, net: MslRay):
        super().__init__()
        self.net = net

    def forward(self, encoded: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        raw = self.net.net(encoded)
        colors, alphas = self.net.rendering.raw2color(raw, depths)
        return torch.cat([colors, alphas[..., None]], -1)
