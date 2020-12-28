from typing import List, Tuple
import torch
import torch.nn as nn
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


class Rendering(nn.Module):

    def __init__(self):
        """
        Initialize a Rendering module
        """
        super().__init__()

    def forward(self, color_alpha: torch.Tensor) -> torch.Tensor:
        """
        Blend layers to get final color

        :param color_alpha ```Tensor(B, L, C)```: RGB or gray with alpha channel
        :return ```Tensor(B, C-1)``` blended pixels
        """
        c = color_alpha[..., :-1]
        a = color_alpha[..., -1:]
        blended = c[:, 0, :] * a[:, 0, :]
        for l in range(1, color_alpha.size(1)):
            blended = blended * (1 - a[:, l, :]) + c[:, l, :] * a[:, l, :]
        return blended


class MslNet(nn.Module):

    def __init__(self, cam_params, fc_params, sphere_layers: List[float],
                 out_res: Tuple[int, int], gray=False, encode_to_dim: int = 0):
        """
        Initialize a multi-sphere-layer net

        :param cam_params: intrinsic parameters of camera
        :param fc_params: parameters of full-connection network
        :param sphere_layers: list(L), radius of sphere layers
        :param out_res: resolution of output view image
        :param gray: is grayscale mode
        :param encode_to_dim: encode input to number of dimensions
        """
        super().__init__()
        self.cam_params = cam_params
        self.sphere_layers = torch.tensor(sphere_layers,
                                          dtype=torch.float,
                                          device=device.GetDevice())
        self.in_chns = 3
        self.out_res = out_res
        self.input_encoder = net_modules.InputEncoder.Get(
            encode_to_dim, self.in_chns)
        fc_params['in_chns'] = self.input_encoder.out_dim
        fc_params['out_chns'] = 2 if gray else 4
        self.net = net_modules.FcNet(**fc_params)
        self.rendering = Rendering()

    def forward(self, ray_positions: torch.Tensor, ray_directions: torch.Tensor) -> torch.Tensor:
        """
        rays -> colors

        :param ray_positions ```Tensor(B, M, 3)|Tensor(B, 3)```: ray positions
        :param ray_directions ```Tensor(B, M, 3)|Tensor(B, 3)```: ray directions
        :return: Tensor(B, 1|3, H, W)|Tensor(B, 1|3), inferred images/pixels
        """
        p = ray_positions.view(-1, 3)
        v = ray_directions.view(-1, 3)
        spher = RayToSpherical(p, v, self.sphere_layers).flatten(0, 1)
        color_alpha = self.net(self.input_encoder(spher)).view(
            p.size(0), self.sphere_layers.size(0), -1)
        c: torch.Tensor = self.rendering(color_alpha)
        # unflatten
        return c.view(ray_directions.size(0), self.out_res[0],
                      self.out_res[1], -1).permute(0, 3, 1, 2) if len(ray_directions.size()) == 3 else c
