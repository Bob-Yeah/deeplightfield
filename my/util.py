from typing import List, Tuple
from math import pi
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import glm
import os

from torchvision.utils import save_image

gvec_type = [glm.dvec1, glm.dvec2, glm.dvec3, glm.dvec4]
gmat_type = [[glm.dmat2, glm.dmat2x3, glm.dmat2x4],
             [glm.dmat3x2, glm.dmat3, glm.dmat3x4],
             [glm.dmat4x2, glm.dmat4x3, glm.dmat4]]


def Fov2Length(angle):
    return np.tan(angle * np.pi / 360) * 2


def SmoothStep(x0, x1, x):
    y = torch.clamp((x - x0) / (x1 - x0), 0, 1)
    return y * y * (3 - 2 * y)


def MatImg2Tensor(img, permute=True, batch_dim=True):
    batch_input = len(img.shape) == 4
    if permute:
        t = torch.from_numpy(np.transpose(img,
                                          [0, 3, 1, 2] if batch_input else [2, 0, 1]))
    else:
        t = torch.from_numpy(img)
    if not batch_input and batch_dim:
        t = t.unsqueeze(0)
    return t


def MatImg2Numpy(img, permute=True, batch_dim=True):
    batch_input = len(img.shape) == 4
    if permute:
        t = np.transpose(img, [0, 3, 1, 2] if batch_input else [2, 0, 1])
    else:
        t = img
    if not batch_input and batch_dim:
        t = t.unsqueeze(0)
    return t


def Tensor2MatImg(t: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor to numpy ndarray suitable for matplotlib

    :param t: 2D (HW), 3D (CHW/HWC) or 4D (BCHW/BHWC) tensor
    :return: numpy ndarray (...C), with channel transposed to the last dim
    """
    img = t.squeeze().cpu().detach().numpy()
    if len(img.shape) == 2:  # Single channel image
        return img
    batch_input = len(img.shape) == 4
    if t.size()[batch_input] <= 4:
        return np.transpose(img, [0, 2, 3, 1] if batch_input else [1, 2, 0])
    return img


def ReadImageTensor(path, permute=True, rgb_only=True, batch_dim=True):
    channels = 3 if rgb_only else 4
    if isinstance(path, list):
        first_image = plt.imread(path[0])[:, :, 0:channels]
        b_image = np.empty(
            (len(path), first_image.shape[0], first_image.shape[1], channels), dtype=np.float32)
        b_image[0] = first_image
        for i in range(1, len(path)):
            b_image[i] = plt.imread(path[i])[:, :, 0:channels]
        return MatImg2Tensor(b_image, permute)
    return MatImg2Tensor(plt.imread(path)[:, :, 0:channels], permute, batch_dim)


def ReadImageNumpyArray(path, permute=True, rgb_only=True, batch_dim=True):
    channels = 3 if rgb_only else 4
    if isinstance(path, list):
        first_image = plt.imread(path[0])[:, :, 0:channels]
        b_image = np.empty(
            (len(path), first_image.shape[0], first_image.shape[1], channels), dtype=np.float32)
        b_image[0] = first_image
        for i in range(1, len(path)):
            b_image[i] = plt.imread(path[i])[:, :, 0:channels]
        return MatImg2Numpy(b_image, permute)
    return MatImg2Numpy(plt.imread(path)[:, :, 0:channels], permute, batch_dim)


def WriteImageTensor(t, path):
    #image = Tensor2MatImg(t)
    if isinstance(path, list):
        if (len(t.size()) != 4 and len(path) != 1) or t.size()[0] != len(path):
            raise ValueError
        for i in range(len(path)):
            save_image(t[i], path[i])
            #plt.imsave(path[i], image[i])
    else:
        if len(t.squeeze().size()) >= 4:
            raise ValueError
        #plt.imsave(path, image)
        save_image(t, path)


def PlotImageTensor(t):
    plt.imshow(Tensor2MatImg(t))


def Tensor2Glm(t):
    t = t.squeeze()
    size = t.size()
    if len(size) == 1:
        if size[0] <= 0 or size[0] > 4:
            raise ValueError
        return gvec_type[size[0] - 1](t.cpu().numpy())
    if len(size) == 2:
        if size[0] <= 1 or size[0] > 4 or size[1] <= 1 or size[1] > 4:
            raise ValueError
        return gmat_type[size[1] - 2][size[0] - 2](t.cpu().numpy())
    raise ValueError


def Glm2Tensor(val):
    return torch.from_numpy(np.array(val))


def MeshGrid(size: Tuple[int, int], normalize: bool = False, swap_dim: bool = False):
    """
    Generate a mesh grid

    :param size: grid size (rows, columns)
    :param normalize: return coords in normalized space? defaults to False
    :param swap_dim: if True, return coords in (y, x) order, defaults to False
    :return: rows x columns x 2 tensor
    """
    y, x = torch.meshgrid(torch.tensor(range(size[0])),
                          torch.tensor(range(size[1])))
    if swap_dim:
        if normalize:
            return torch.stack([y / (size[0] - 1.), x / (size[1] - 1.)], 2)
        else:
            return torch.stack([y, x], 2)
    if normalize:
        return torch.stack([x / (size[1] - 1.), y / (size[0] - 1.)], 2)
    else:
        return torch.stack([x, y], 2)


def CreateDirIfNeed(path):
    if not os.path.exists(path):
        os.makedirs(path)


def GetLocalViewRays(cam_params, res: Tuple[int, int], flatten=False) -> torch.Tensor:
    coords = MeshGrid(res)
    c = torch.tensor([cam_params['cx'], cam_params['cy']])
    f = torch.tensor([cam_params['fx'], cam_params['fy']])
    rays = torch.cat([
        (coords - c) / f,
        torch.ones(res[0], res[1], 1, )
    ], dim=2)
    if flatten:
        rays = rays.flatten(0, 1)
    return rays


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


def GetDepthLayers(depth_range: Tuple[float, float], n_layers: int) -> List[float]:
    """
    Get [n_layers] foreground layers whose diopters are distributed uniformly
    in  [depth_range] plus a background layer

    :param depth_range: depth range of foreground layers
    :param n_layers: number of foreground layers
    :return: list of [n_layers+1] depths
    """
    diopter_range = (1 / depth_range[1], 1 / depth_range[0])
    depths = [1e5]  # Background layer
    depths += list(1.0 / np.linspace(diopter_range[0], diopter_range[1], n_layers))
    return depths
