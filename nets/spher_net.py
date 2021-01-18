import torch
import torch.nn as nn
from ..my import net_modules
from ..my import util


class SpherNet(nn.Module):

    def __init__(self, fc_params,
                 gray: bool = False,
                 translation: bool = False,
                 encode_to_dim: int = 0):
        """
        Initialize a sphere net

        :param fc_params: parameters for full-connection network
        :param gray: whether grayscale mode
        :param translation: whether support translation of view
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.in_chns = 5 if translation else 2
        self.input_encoder = net_modules.InputEncoder.Get(
            encode_to_dim, self.in_chns)
        fc_params['in_chns'] = self.input_encoder.out_dim
        fc_params['out_chns'] = 1 if gray else 3
        self.net = net_modules.FcNet(**fc_params)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        """
        rays -> colors

        :param rays_o ```Tensor(B, ..., 3)```: rays' origin
        :param rays_d ```Tensor(B, ..., 3)```: rays' direction
        :return: Tensor(B, 1|3, ...), inferred images/pixels
       """
        p = rays_o.view(-1, 3)
        v = rays_d.view(-1, 3)
        spher = util.CartesianToSpherical(v)[..., 1:3]  # (..., 2)
        input = torch.cat([p, spher], dim=-1) if self.in_chns == 5 else spher

        c: torch.Tensor = self.net(self.input_encoder(input))

        # Unflatten according to input shape
        out_shape = list(rays_d.size())
        out_shape[-1] = -1
        return c.view(out_shape).movedim(-1, 1)