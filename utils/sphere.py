from typing import List, Union
import torch
import math
from . import misc


def cartesian2spherical(cart: torch.Tensor, inverse_r: bool = False) -> torch.Tensor:
    """
    Convert coordinates from Cartesian to Spherical

    :param cart `Tensor(..., 3)`: coordinates in Cartesian
    :param inverse_r: whether to inverse r
    :return `Tensor(..., 3)`: coordinates in Spherical (r, theta, phi)
    """
    rho = torch.sqrt(torch.sum(cart * cart, dim=-1))
    theta = misc.get_angle(cart[..., 0], cart[..., 2])
    if inverse_r:
        rho = rho.reciprocal()
        phi = torch.acos(cart[..., 1] * rho)
    else:
        phi = torch.acos(cart[..., 1] / rho)
    return torch.stack([rho, theta, phi], dim=-1)


def spherical2cartesian(spher: torch.Tensor) -> torch.Tensor:
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


def ray_sphere_intersect(p: torch.Tensor, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Calculate intersections of each rays and each spheres

    :param p `Tensor(B, 3)`: positions of rays
    :param v `Tensor(B, 3)`: directions of rays
    :param r `Tensor(N)`: , radius of spheres
    :return `Tensor(B, N, 3)`: points of intersection
    :return `Tensor(B, N)`: depths of intersection along ray
    """
    # p, v: Expand to (B, 1, 3)
    p = p.unsqueeze(1)
    v = v.unsqueeze(1)
    # pp, vv, pv: (B, 1)
    pp = (p * p).sum(dim=2)
    vv = (v * v).sum(dim=2)
    pv = (p * v).sum(dim=2)
    depths = (((pv * pv - vv * (pp - r * r)).sqrt() - pv) / vv)
    return p + depths[..., None] * v, depths


def get_rot_matrix(theta: Union[float, torch.Tensor], phi: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Get rotation matrix from angles in spherical space

    :param theta `Tensor(..., 1) | float`: rotation angles around y axis
    :param phi  `Tensor(..., 1) | float`: rotation angles around x axis
    :return: `Tensor(..., 3, 3)` rotation matrices
    """
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor([theta])
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor([phi])
    spher = torch.cat([torch.ones_like(theta), theta, phi], dim=-1)
    print(spher)
    forward = spherical2cartesian(spher)  # (..., 3)
    up = torch.tensor([0.0, 1.0, 0.0])
    forward, up = torch.broadcast_tensors(forward, up)
    print(forward, up)
    right = torch.cross(forward, up, dim=-1)  # (..., 3)
    up = torch.cross(right, forward, dim=-1)  # (..., 3)
    print(right, up, forward)
    return torch.stack([right, up, forward], dim=-2)  # (..., 3, 3)


def calc_local_dir(dirs, spherical_coords, pts):
    """
    [summary]

    :param dirs `Tensor(B, 3)`: 
    :param spherical_coords `Tensor(B, N, 3)`: 
    :param pts `Tensor(B, N, 3)`: 
    :return `Tensor(B, N, 2)`
    """
    local_z = pts / pts.norm(dim=-1, keepdim=True)
    local_x = spherical2cartesian(
        spherical_coords + torch.tensor([0, math.radians(0.1), 0], device=spherical_coords.device)) - pts
    local_x = local_x / local_x.norm(dim=-1, keepdim=True)
    local_y = torch.cross(local_x, local_z, -1)
    local_rot = torch.stack([local_x, local_y, local_z], dim=-2)  # (B, N, 3, 3)
    return cartesian2spherical(torch.matmul(dirs[:, None, None, :], local_rot)) \
        .squeeze(-2)[..., 1:3]
