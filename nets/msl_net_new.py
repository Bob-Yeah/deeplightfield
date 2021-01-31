import torch
import torch.nn as nn
from .modules import *
from ..my import color_mode
from ..my.simple_perf import SimplePerf


class NewMslNet(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 normalize_coord: bool,
                 dir_as_input: bool,
                 n_nets: int = 2,
                 not_same_net: bool = False,
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

        if not_same_net:
            self.n_nets = 2
            self.nets = nn.ModuleList([
                FcNet(**fc_params),
                FcNet(in_chns=fc_params['in_chns'],
                      out_chns=fc_params['out_chns'],
                      nf=128, n_layers=4)
            ])
        else:
            self.n_nets = n_nets
            self.nets = nn.ModuleList([
                FcNet(**fc_params) for _ in range(n_nets)
            ])
        self.n_samples = sampler_params['n_samples']

    def update_normalize_range(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        coords, _, _ = self.sampler(rays_o, rays_d)
        coords = coords[..., 1:].view(-1, 2)
        self.angle_range = torch.stack([
            torch.cat([coords, self.angle_range[0:1]]).amin(0),
            torch.cat([coords, self.angle_range[1:2]]).amax(0)
        ])

    def sample_and_infer(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                         sampler: Sampler = None) -> torch.Tensor:
        if not sampler:
            sampler = self.sampler
        coords, pts, depths = sampler(rays_o, rays_d)

        encoded = self.input_encoder(coords)

        sn = sampler.samples // self.n_nets
        raw = torch.cat([
            self.nets[i](encoded[:, i * sn:(i + 1) * sn])
            for i in range(self.n_nets)
        ], 1)
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
