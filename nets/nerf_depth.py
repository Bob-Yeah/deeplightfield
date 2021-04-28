import torch
import torch.nn as nn
from .modules import *
from utils import color

'''
The first step towards depth-guide acceleration
Sample according to raw depth input
'''


class NerfDepth(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 c: int = color.RGB,
                 coord_encode: int = 0,
                 n_bins: int = 128,
                 include_neighbor_bins=True):
        super().__init__()
        self.color = c
        self.n_samples = sampler_params['n_samples']
        self.coord_chns = 3
        self.color_chns = color.chns(self.color)
        self.coord_encoder = InputEncoder.Get(coord_encode, self.coord_chns)
        self.mlp = NewMlp(coord_chns=self.coord_encoder.out_dim,
                          density_chns=1,
                          color_chns=self.color_chns,
                          core_nf=fc_params['nf'],
                          core_layers=fc_params['n_layers'],
                          activation=fc_params['activation'],
                          skips=fc_params['skips'])
        self.sampler = AdaptiveSampler(**sampler_params, n_bins=n_bins,
                                       include_neighbor_bins=include_neighbor_bins)
        self.rendering = NewRendering()

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                rays_depth: torch.Tensor, rays_bins: torch.Tensor,
                ret_depth=False, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param rays_depth `Tensor(B)`: rays' depth
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        coords, pts, depths, _ = self.sampler(rays_o, rays_d, rays_depth, rays_bins)
        encoded_position = self.coord_encoder(coords)
        colors, densities = self.mlp(encoded_position)
        return self.rendering(colors, densities[..., 0], depths,
                              ret_depth=ret_depth, debug=debug)
