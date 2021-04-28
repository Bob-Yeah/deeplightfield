import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_f
from .modules import *
from utils import sphere
from utils import color


class CNerf(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 c: int = color.RGB,
                 coord_encode: int = 0):
        super().__init__()
        self.color = c
        self.n_samples = sampler_params['n_samples']
        self.coord_chns = 3
        self.color_chns = color.chns(self.color)
        self.coord_encoder = InputEncoder.Get(coord_encode, self.coord_chns)
        self.density_net = Mlp(coord_chns=self.coord_encoder.out_dim, density_chns=1, color_chns=0,
                               core_nf=fc_params['nf'], core_layers=fc_params['n_layers'])
        self.color_net = Mlp(coord_chns=self.coord_encoder.out_dim * self.n_samples,
                             density_chns=0, color_chns=self.color_chns * self.n_samples,
                             core_nf=fc_params['nf'], core_layers=fc_params['n_layers'])
        self.sampler = Sampler(**sampler_params)
        self.rendering = NewRendering()

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                ret_depth=False, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        coords, pts, depths = self.sampler(rays_o, rays_d)
        encoded_position = self.coord_encoder(coords)
        densities = self.density_net(encoded_position)[1][..., 0]
        colors = self.color_net(encoded_position.flatten(1, 2))[0]. \
            view(-1, self.n_samples, self.color_chns)
        return self.rendering(colors, densities, depths, ret_depth=ret_depth, debug=debug)
