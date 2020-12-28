from typing import List, Tuple
from math import pi
import torch
import torch.nn as nn
from .pytorch_prototyping.pytorch_prototyping import *
from .my import net_modules
from .my import util
from .my import device


def RaySphereIntersect(p: torch.Tensor, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Calculate intersections of each rays and each spheres

    :param p: B x 3, positions of rays
    :param v: B x 3, directions of rays
    :param r: B'(1D), radius of spheres
    :return: B x B' x 3, points of intersection
    """
    # p, v: Expand to B x 1 x 3
    p = p.unsqueeze(1)
    v = v.unsqueeze(1)
    # pp, vv, pv: B x 1
    pp = (p * p).sum(dim=2)
    vv = (v * v).sum(dim=2)
    pv = (p * v).sum(dim=2)
    # k: Expand to B x B' x 1
    k = (((pv * pv - vv * (pp - r * r)).sqrt() - pv) / vv).unsqueeze(2)
    return p + k * v


def RayToSpherical(p: torch.Tensor, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Calculate intersections of each rays and each spheres

    :param p: B x 3, positions of rays
    :param v: B x 3, directions of rays
    :param r: B' x 1, radius of spheres
    :return: B x B' x 3, spherical coordinates
    """
    p_on_spheres = RaySphereIntersect(p, v, r)
    return util.CartesianToSpherical(p_on_spheres)


class SpherNet(nn.Module):

    def __init__(self, cam_params,  # spher_min: Tuple[float, float], spher_max: Tuple[float, float],
                 fc_params,
                 out_res: Tuple[int, int] = None,
                 gray: bool = False,
                 encode_to_dim: int = 0):
        """
        Initialize a sphere net

        :param cam_params: intrinsic parameters of camera
        :param fc_params: parameters of full-connection network
        :param out_res: resolution of output view image
        :param gray: is grayscale mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.cam_params = cam_params
        self.in_chns = 2
        self.out_res = out_res
        #self.spher_min = torch.tensor(spher_min, device=device.GetDevice()).view(1, 2)
        #self.spher_max = torch.tensor(spher_max, device=device.GetDevice()).view(1, 2)
        #self.spher_range = self.spher_max - self.spher_min
        self.input_encoder = net_modules.InputEncoder.Get(encode_to_dim, self.in_chns)
        fc_params['in_chns'] = self.input_encoder.out_dim
        fc_params['out_chns'] = 1 if gray else 3
        self.net = net_modules.FcNet(**fc_params)

    def forward(self, _, ray_directions: torch.Tensor) -> torch.Tensor:
        """
        rays -> colors

        :param ray_directions ```Tensor(B, M, 3)|Tensor(B, 3)```: ray directions
        :return: Tensor(B, 1|3, H, W)|Tensor(B, 1|3), inferred images/pixels
        """
        v = ray_directions.view(-1, 3)  # (*, 3)
        spher = util.CartesianToSpherical(v)[..., 1:3]  # (*, 2)
        # (spher - self.spher_min) / self.spher_range * 2 - 0.5
        spher_normed = spher

        c: torch.Tensor = self.net(self.input_encoder(spher_normed))
        # Unflatten to (B, 1|3, H, W) if take view as item
        return c.view(ray_directions.size(0), self.out_res[0], self.out_res[1],
                      -1).permute(0, 3, 1, 2) if len(ray_directions.size()) == 3 else c
