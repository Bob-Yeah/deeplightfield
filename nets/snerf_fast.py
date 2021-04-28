import math
import torch
import torch.nn as nn
from torch.nn.modules import module
from .modules import *
from utils import sphere
from utils import color


class SnerfFast(nn.Module):

    def __init__(self, fc_params, sampler_params, n_parts: int, normalize_coord: bool,
                 c: int = color.RGB, coord_encode: int = 0, dir_encode: int = None,
                 spherical_dir: bool = False, multiple_net=True):
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
        self.n_parts = n_parts
        self.samples_per_part = self.n_samples // self.n_parts
        self.coord_chns = 2 if multiple_net else 3
        self.color_chns = color.chns(self.color)
        self.coord_encoder = InputEncoder.Get(coord_encode, self.coord_chns)

        if dir_encode is not None:
            self.dir_encoder = InputEncoder.Get(dir_encode, 2 if self.spherical_dir else 3)
            self.dir_chns_per_part = self.dir_encoder.out_dim * \
                (self.samples_per_part if self.spherical_dir else 1)
        else:
            self.dir_encoder = None
            self.dir_chns_per_part = 0

        if multiple_net:
            old = False
            if old:
                self.mlp = Mlp(coord_chns=self.coord_encoder.out_dim * self.samples_per_part,
                               density_chns=self.samples_per_part,
                               color_chns=self.color_chns * self.samples_per_part,
                               core_nf=fc_params['nf'],
                               core_layers=fc_params['n_layers'],
                               dir_chns=self.dir_chns_per_part,
                               dir_nf=fc_params['nf'] // 2,
                               activation=fc_params['activation'])
                self.nets = [self.mlp]
                if n_parts > 1:
                    self.nets += [
                        Mlp(coord_chns=self.coord_encoder.out_dim * self.samples_per_part,
                            density_chns=self.samples_per_part,
                            color_chns=self.color_chns * self.samples_per_part,
                            core_nf=fc_params['nf'],
                            core_layers=fc_params['n_layers'],
                            dir_chns=self.dir_chns_per_part,
                            dir_nf=fc_params['nf'] // 2,
                            activation=fc_params['activation'])
                        for _ in range(1, self.n_parts)
                    ]
                    for i in range(1, self.n_parts):
                        self.add_module(f"mlp_{i:d}", self.nets[i])
            else:
                self.nets = [
                    NewMlp(coord_chns=self.coord_encoder.out_dim * self.samples_per_part,
                           density_chns=self.samples_per_part,
                           color_chns=self.color_chns * self.samples_per_part,
                           core_nf=fc_params['nf'],
                           core_layers=fc_params['n_layers'],
                           dir_chns=self.dir_chns_per_part,
                           dir_nf=fc_params['nf'] // 2,
                           activation=fc_params['activation'])
                    for _ in range(self.n_parts)
                ]
                for i in range(self.n_parts):
                    self.add_module(f"mlp_{i:d}", self.nets[i])
        else:
            self.nets = None
            self.net = Mlp(coord_chns=self.coord_encoder.out_dim * self.samples_per_part,
                           density_chns=self.samples_per_part,
                           color_chns=self.color_chns * self.samples_per_part,
                           core_nf=fc_params['nf'],
                           core_layers=fc_params['n_layers'],
                           dir_chns=self.dir_chns_per_part,
                           dir_nf=fc_params['nf'] // 2,
                           activation=fc_params['activation'])
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
        coords_encoded = self.coord_encoder(coords[..., -self.coord_chns:])
        dirs_encoded = self.dir_encoder(
            sphere.calc_local_dir(rays_d, coords, pts) if self.spherical_dir else rays_d) \
            if self.dir_encoder is not None else None
        densities = torch.empty(rays_o.size(0), self.n_samples, device=device.default())
        colors = torch.empty(rays_o.size(0), self.n_samples, self.color_chns,
                             device=device.default())
        if self.nets is not None:
            for i, net in enumerate(self.nets):
                s = slice(i * self.samples_per_part, (i + 1) * self.samples_per_part)
                c, d = net(coords_encoded[:, s].flatten(1, 2),
                           dirs_encoded[:, s].flatten(1, 2) if self.spherical_dir else dirs_encoded)
                colors[:, s] = c.view(-1, self.samples_per_part, self.color_chns)
                densities[:, s] = d
        else:
            for i in range(self.n_parts):
                s = slice(i * self.samples_per_part, (i + 1) * self.samples_per_part)
                c, d = self.net(coords_encoded[:, s].flatten(1, 2),
                                dirs_encoded[:, s].flatten(1, 2) if self.spherical_dir else dirs_encoded)
                colors[:, s] = c.view(-1, self.samples_per_part, self.color_chns)
                densities[:, s] = d
        densities = densities
        return self.rendering(colors.view(-1, self.n_samples, self.color_chns),
                              densities, depths, ret_depth=ret_depth, debug=debug)


class SnerfFastExport(nn.Module):

    def __init__(self, net: SnerfFast):
        super().__init__()
        self.net = net

    def forward(self, coords_encoded, z_vals):
        colors = []
        densities = []
        for i in range(self.net.n_parts):
            s = slice(i * self.net.samples_per_part, (i + 1) * self.net.samples_per_part)
            mlp = self.net.nets[i] if self.net.nets is not None else self.net.net
            c, d = mlp(coords_encoded[:, s].flatten(1, 2))
            colors.append(c.view(-1, self.net.samples_per_part, self.net.color_chns))
            densities.append(d)
        colors = torch.cat(colors, 1)
        densities = torch.cat(densities, 1)
        alphas = self.net.rendering.density2alpha(densities, z_vals)
        return torch.cat([colors, alphas[..., None]], -1)
