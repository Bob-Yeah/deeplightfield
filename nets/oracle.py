import torch
import torch.nn as nn
from .modules import *


class Oracle(nn.Module):

    def __init__(self, fc_params, sampler_params, normalize_coord: bool,
                 coord_encode: int = 0, spherical_dir: bool = False, out_activation='sigmoid'):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param normalize_coord: whether normalize the spherical coords to [0, 2pi] before encode
        :param c: color mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.normalize_coord = normalize_coord
        self.spherical_dir = spherical_dir
        self.n_samples = sampler_params['n_samples']
        self.coord_chns = 3
        self.sampler = Sampler(**sampler_params)
        self.renderer = NewRendering()
        self.coord_encoder = InputEncoder.Get(coord_encode, self.coord_chns)
        self.net = nn.Sequential(
            FcNet(in_chns=self.coord_encoder.out_dim * self.n_samples,
                    out_chns=0, nf=fc_params['nf'], n_layers=fc_params['n_layers'],
                    skips=[], activation=fc_params['activation']),
            FcLayer(fc_params['nf'], self.n_samples, out_activation)
        )

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        coords, _, z_vals = self.sampler(rays_o, rays_d)
        coords_encoded = self.coord_encoder(coords)
        return self.net(coords_encoded.flatten(1, 2))