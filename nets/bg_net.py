import torch
import torch.nn as nn
from .modules import *
from utils import sphere
from utils import color


class BgNet(nn.Module):

    def __init__(self, fc_params, encode, c):
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
        self.coord_chns = 2
        self.color_chns = color.chns(self.color)

        self.coord_encoder = InputEncoder.Get(encode, self.coord_chns)
        self.mlp = Mlp(coord_chns=self.coord_encoder.out_dim,
                       density_chns=0,
                       color_chns=self.color_chns,
                       core_nf=fc_params['nf'],
                       core_layers=fc_params['n_layers'],
                       activation=fc_params['activation'])

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return: 'color' -> `Tensor(B, C)``, inferred colors
        """
        coords_encoded = self.coord_encoder(sphere.cartesian2spherical(rays_d)[..., 1:])
        return {'color': self.mlp(coords_encoded)[0]}