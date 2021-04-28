import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_f
from .modules import *
from utils import sphere
from utils import color
from itertools import product

'''
The first step towards depth-guide acceleration
Sample according to raw depth input
'''


class CNerf(nn.Module):

    def __init__(self, fc_params, sampler_params,
                 c: int = color.RGB,
                 coord_encode: int = 0,
                 n_bins: int = 128):
        super().__init__()
        self.color = c
        self.n_samples = sampler_params['n_samples']
        self.n_bins = n_bins
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
        self.sampler = PdfSampler(**sampler_params, n_bins=n_bins)
        self.rendering = NewRendering()

    def set_depth_maps(self, rays_o, rays_d, depthmaps):
        """
        [summary]

        :param rays_o `Tensor(B, H, W, 3)`
        :param rays_d `Tensor(B, H, W, 3)
        :param depthmaps `Tensor(B, H, W)`: [description]
        """
        with torch.no_grad():
            radius_maps = sphere.cartesian2spherical(rays_o + rays_d * depthmaps[..., None],
                                                     inverse_r=self.sampler.lindisp)[..., 0]
            bin_ids = torch.floor((radius_maps - self.sampler.s_range[0]) /
                                  (self.sampler.s_range[1] - self.sampler.s_range[0]) * self.n_bins)
            bin_ids = bin_ids.clamp(0, self.n_bins - 1).to(torch.long)[..., None]

            k = 3
            self.bin_weights = torch.zeros_like(bin_ids.expand(-1, -1, -1, self.n_bins),
                                                dtype=torch.int8)  # (B, H, W, N)
            # 10 Views per batch to keep memory cost low enough
            batch_size = 10
            temp_weights = torch.empty_like(self.bin_weights[:batch_size])  # (B', H, W, N)
            for offset in range(0, bin_ids.size(0), 10):
                bidx = slice(offset, min(offset + batch_size, bin_ids.size(0)))
                idx = slice(0, bidx.stop - bidx.start)
                temp_weights.fill_(0)
                for i, j in product(range(-2, 3), range(-2, 3)):
                    w = int(10 * (1 - math.sqrt(0.5 * (i * i + j * j)) / 3))
                    src_sy = slice(-j) if j < 0 else slice(j, None)
                    src_sx = slice(-i) if i < 0 else slice(i, None)
                    dst_sy = slice(-j) if j > 0 else slice(j, None)
                    dst_sx = slice(-i) if i > 0 else slice(i, None)
                    bin_ids_subview = bin_ids[bidx, src_sy, src_sx]
                    weights_subview = temp_weights[idx, dst_sy, dst_sx]
                    weights_subview.scatter_(-1, bin_ids_subview,
                                             weights_subview.gather(-1, bin_ids_subview).clamp_min(w))
                # Only keep top-k bins
                _, bin_idxs = torch.topk(temp_weights[idx], k)  # (B', H, W, N)
                self.bin_weights[bidx].scatter_(-1, bin_idxs, 1)
                #                                 temp_depth_weights[idx].gather(-1, bin_idxs))

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, rays_weights: torch.Tensor,
                ret_depth=False, debug=False) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param rays_depth `Tensor(B)`: rays' depth
        :return: `Tensor(B, C)``, inferred images/pixels
        """
        coords, pts, depths, _ = self.sampler(rays_o, rays_d, rays_weights)
        encoded_position = self.coord_encoder(coords)
        colors, densities = self.mlp(encoded_position)
        return self.rendering(colors, densities[..., 0], depths,
                              ret_depth=ret_depth, debug=debug)
