import math
import torch
import torch.nn as nn
from .modules import *
from my import util
from my import color_mode

class MslNet(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 normalize_coord: bool,
                 dir_as_input: bool,
                 color: int = color_mode.RGB,
                 encode_to_dim: int = 0,
                 export_mode: bool = False):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param normalize_coord: whether normalize the spherical coords to [0, 2pi] before encode
        :param color: color mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.in_chns = 3
        self.input_encoder = InputEncoder.Get(
            encode_to_dim, self.in_chns)
        fc_params['in_chns'] = self.input_encoder.out_dim
        fc_params['out_chns'] = 2 if color == color_mode.GRAY else 4
        self.sampler = Sampler(**sampler_params)
        self.rendering = Rendering()
        self.export_mode = export_mode
        self.normalize_coord = normalize_coord
        self.dir_as_input = dir_as_input
        self.color = color
        if self.color == color_mode.YCbCr:
            self.net1 = FcNet(
                in_chns=fc_params['in_chns'],
                out_chns=fc_params['nf'] + 2,
                nf=fc_params['nf'],
                n_layers=fc_params['n_layers'] - 2)
            self.net2 = FcNet(
                in_chns=fc_params['nf'],
                out_chns=2,
                nf=fc_params['nf'],
                n_layers=1)
            self.net = None
        elif self.dir_as_input:
            self.input_encoder2 = InputEncoder.Get(4, 2)
            self.net1 = FcNet(
                in_chns=fc_params['in_chns'],
                out_chns=fc_params['nf'],
                nf=fc_params['nf'],
                n_layers=fc_params['n_layers'])
            self.net2 = FcNet(
                in_chns=fc_params['nf'] + self.input_encoder2.out_dim,
                out_chns=fc_params['out_chns'],
                nf=fc_params['nf'],
                n_layers=1)
            self.net = None
        else:
            self.net = FcNet(**fc_params)
        if self.normalize_coord:
            self.register_buffer('angle_range', torch.tensor(
                [[1e5, 1e5], [-1e5, -1e5]]))
            self.register_buffer('depth_range', torch.tensor([
                self.sampler.lower[0], self.sampler.upper[-1]
            ]))
        self.n_samples = sampler_params['n_samples']

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

        :param rays_d ```Tensor(B, 3)```: 
        :param coords ```Tensor(B, N, 3)```: 
        :param pts ```Tensor(B, N, 3)```: 
        :return ```Tensor(B, N, 2)```
        """
        local_z = pts / pts.norm(dim=-1, keepdim=True)
        local_x = util.SphericalToCartesian(
            coords + torch.tensor([0, 0.1 / 180 * math.pi, 0], device=coords.device)) - pts
        local_x = local_x / local_x.norm(dim=-1, keepdim=True)
        local_y = torch.cross(local_x, local_z, -1)
        local_rot = torch.stack(
            [local_x, local_y, local_z], dim=-2)  # (B, N, 3, 3)
        return util.CartesianToSpherical(torch.matmul(
            rays_d[:, None, None, :], local_rot)).squeeze(-2)[..., 1:3]

    def sample_and_infer(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                         sampler: Sampler = None) -> torch.Tensor:
        if not sampler:
            sampler = self.sampler
        coords, pts, depths = sampler(rays_o, rays_d)

        if self.dir_as_input:
            dirs = self.calc_local_dir(rays_d, coords, pts)

        if self.normalize_coord:  # Normalize coords to [0, 2pi]
            range = torch.cat(
                [self.depth_range.view(2, 1), self.angle_range], 1)
            coords = (coords - range[0]) / (range[1] - range[0]) * 2 * math.pi
        encoded = self.input_encoder(coords)

        if self.color == color_mode.YCbCr:
            mid_output = self.net1(encoded)
            net2_output = self.net2(mid_output[..., :-2])
            raw = torch.cat([
                mid_output[..., -2:],
                net2_output
            ], -1)
        elif self.dir_as_input:
            encoded_dirs = self.input_encoder2(dirs)
            #print(encoded.size(), self.net1(encoded).size(), encoded_dirs.size())
            raw = self.net2(torch.cat([self.net1(encoded), encoded_dirs], -1))
        else:
            raw = self.net(encoded)
        return raw, depths

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                ret_depth: bool = False, sampler: Sampler = None) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o ```Tensor(B, 3)```: rays' origin
        :param rays_d ```Tensor(B, 3)```: rays' direction
        :return: ```Tensor(B, C)``, inferred images/pixels
        """
        raw, depths = self.sample_and_infer(rays_o, rays_d, sampler)
        if self.export_mode:
            colors, alphas = self.rendering.raw2color(raw, depths)
            return torch.cat([colors, alphas[..., None]], -1)

        if ret_depth:
            color_map, _, _, _, depth_map = self.rendering(
                raw, depths, ret_extra=True)
            return color_map, depth_map

        return self.rendering(raw, depths)


class ExportNet(nn.Module):

    def __init__(self, net: MslNet):
        super().__init__()
        self.net = net

    def forward(self, encoded: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        raw = self.net.net(encoded)
        colors, alphas = self.net.rendering.raw2color(raw, depths)
        return torch.cat([colors, alphas[..., None]], -1)
