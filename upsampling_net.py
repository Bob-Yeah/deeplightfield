from typing import Tuple
import torch
import torch.nn as nn
from .my import net_modules
from .my import util
from .my import device



class UpsamplingNet(nn.Module):

    def __init__(self, inner_chns, gray=False,
                 encode_to_dim: int = 0):
        """
        Initialize a multi-sphere-layer net

        :param fc_params: parameters for full-connection network
        :param sampler_params: parameters for sampler
        :param gray: is grayscale mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.in_chns = 3
        self.input_encoder = net_modules.InputEncoder.Get(
            encode_to_dim, self.in_chns)
        fc_params['in_chns'] = self.input_encoder.out_dim
        fc_params['out_chns'] = 2 if gray else 4
        self.sampler = Sampler(**sampler_params)
        self.net = net_modules.FcNet(**fc_params)
        self.rendering = Rendering()

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o ```Tensor(B, ..., 3)```: rays' origin
        :param rays_d ```Tensor(B, ..., 3)```: rays' direction
        :return: Tensor(B, 1|3, ...), inferred images/pixels
        """
        p = rays_o.view(-1, 3)
        v = rays_d.view(-1, 3)
        coords, depths = self.sampler(p, v)
        encoded = self.input_encoder(coords)
        color_map = self.rendering(self.net(encoded), depths)
        
        # Unflatten according to input shape
        out_shape = list(rays_d.size())
        out_shape[-1] = -1
        return color_map.view(out_shape).movedim(-1, 1)
