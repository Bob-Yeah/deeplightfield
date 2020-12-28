from typing import List, Tuple
from math import pi
import torch
import torch.nn as nn
from .pytorch_prototyping.pytorch_prototyping import *
from .my import util
from .my import device


def CartesianToSpherical(cart: torch.Tensor) -> torch.Tensor:
    """
    Convert coordinates from Cartesian to Spherical

    :param cart: ... x 3, coordinates in Cartesian
    :return: ... x 3, coordinates in Spherical (r, theta, phi)
    """
    rho = torch.norm(cart, p=2, dim=-1)
    theta = torch.atan2(cart[..., 2], cart[..., 0])
    theta = theta + (theta < 0).type_as(theta) * (2 * pi)
    phi = torch.acos(cart[..., 1] / rho)
    return torch.stack([rho, theta, phi], dim=-1)


def SphericalToCartesian(spher: torch.Tensor) -> torch.Tensor:
    """
    Convert coordinates from Spherical to Cartesian

    :param spher: ... x 3, coordinates in Spherical
    :return: ... x 3, coordinates in Cartesian (r, theta, phi)
    """
    rho = spher[..., 0]
    sin_theta_phi = torch.sin(spher[..., 1:3])
    cos_theta_phi = torch.cos(spher[..., 1:3])
    x = rho * cos_theta_phi[..., 0] * sin_theta_phi[..., 1]
    y = rho * cos_theta_phi[..., 1]
    z = rho * sin_theta_phi[..., 0] * sin_theta_phi[..., 1]
    return torch.stack([x, y, z], dim=-1)


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
    return CartesianToSpherical(p_on_spheres)


class FcNet(nn.Module):

    def __init__(self, in_chns: int, out_chns: int, nf: int, n_layers: int):
        super().__init__()
        self.layers = list()
        self.layers += [
            nn.Linear(in_chns, nf),
            #nn.LayerNorm([nf]),
            nn.ReLU()
        ]
        for _ in range(1, n_layers):
            self.layers += [
                nn.Linear(nf, nf),
                #nn.LayerNorm([nf]),
                nn.ReLU()
            ]
        self.layers.append(nn.Linear(nf, out_chns))
        self.net = nn.Sequential(*self.layers)
        self.net.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)


class Rendering(nn.Module):

    def __init__(self, sphere_layers: List[float]):
        """
        Initialize a Rendering module

        :param sphere_layers: L x 1, radius of sphere layers
        """
        super().__init__()
        self.sphere_layers = torch.tensor(
            sphere_layers, device=device.GetDevice())

    def forward(self, net: FcNet, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        [summary]

        :param net: the full-connected net
        :param p: B x 3, positions of rays
        :param v: B x 3, directions of rays
        :return B x 1/3, view images by blended layers
        """
        L = self.sphere_layers.size()[0]
        sp = RayToSpherical(p, v, self.sphere_layers)  # B x L x 3
        sp[..., 0] = 1 / sp[..., 0]                    # Radius to diopter
        color_alpha: torch.Tensor = net(
            sp.flatten(0, 1)).view(p.size()[0], L, -1)
        if (color_alpha.size(-1) == 2):  # Grayscale
            c = color_alpha[..., 0:1]
            a = color_alpha[..., 1:2]
        else:                           # RGB
            c = color_alpha[..., 0:3]
            a = color_alpha[..., 3:4]
        blended = c[:, 0, :] * a[:, 0, :]
        for l in range(1, L):
            blended = blended * (1 - a[:, l, :]) + c[:, l, :] * a[:, l, :]
        return blended


class MslNet(nn.Module):

    def __init__(self, cam_params, sphere_layers: List[float], out_res: Tuple[int, int], gray=False):
        """
        Initialize a multi-sphere-layer net

        :param cam_params: intrinsic parameters of camera
        :param sphere_layers: L x 1, radius of sphere layers
        :param out_res: resolution of output view image
        """
        super().__init__()
        self.cam_params = cam_params
        self.out_res = out_res
        self.v_local = util.GetLocalViewRays(self.cam_params, out_res, flatten=True) \
            .to(device.GetDevice()) # N x 3
        #self.net = FCBlock(hidden_ch=64,
        #                   num_hidden_layers=4,
        #                   in_features=3,
        #                   out_features=2 if gray else 4,
        #                   outermost_linear=True)
        self.net = FcNet(in_chns=3, out_chns=2 if gray else 4, nf=256, n_layers=8)
        self.rendering = Rendering(sphere_layers)

    def forward(self, view_centers: torch.Tensor, view_rots: torch.Tensor) -> torch.Tensor:
        """
        T_view -> image

        :param view_centers: B x 3, centers of views
        :param view_rots: B x 3 x 3, rotation matrices of views
        :return: B x 1/3 x H_out x W_out, inferred images of views
        """
        # Transpose matrix so we can perform vec x mat
        view_rots_t = view_rots.permute(0, 2, 1)

        # p and v are B x N x 3 tensor
        p = view_centers.unsqueeze(1).expand(-1, self.v_local.size(0), -1)
        v = torch.matmul(self.v_local, view_rots_t)
        c: torch.Tensor = self.rendering(
            self.net, p.flatten(0, 1), v.flatten(0, 1))  # (BN) x 3
        # unflatten
        return c.view(view_centers.size(0), self.out_res[0],
                      self.out_res[1], -1).permute(0, 3, 1, 2)
