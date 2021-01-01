from typing import List, Tuple, Union
import os
import math
import torch
import torchvision
import torchvision.transforms.functional as trans_func
import glm
import numpy as np
import matplotlib.pyplot as plt
from torch.types import Number

from torchvision.utils import save_image

gvec_type = [glm.dvec1, glm.dvec2, glm.dvec3, glm.dvec4]
gmat_type = [[glm.dmat2, glm.dmat2x3, glm.dmat2x4],
             [glm.dmat3x2, glm.dmat3, glm.dmat3x4],
             [glm.dmat4x2, glm.dmat4x3, glm.dmat4]]


def Fov2Length(angle):
    return math.tan(math.radians(angle) / 2) * 2


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


def GetLocalViewRays(cam_params, res: Tuple[int, int], flatten=False, norm=True) -> torch.Tensor:
    coords = MeshGrid(res)
    c = torch.tensor([cam_params['cx'], cam_params['cy']])
    f = torch.tensor([cam_params['fx'], cam_params['fy']])
    rays = broadcast_cat((coords - c) / f, 1.0)
    if norm:
        rays = rays / rays.norm(dim=-1, keepdim=True)
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
    theta = theta + (theta < 0).type_as(theta) * (2 * math.pi)
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
    depths += list(1.0 /
                   np.linspace(diopter_range[0], diopter_range[1], n_layers))
    return depths


def GetRotMatrix(theta: Union[float, torch.Tensor], phi: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Get rotation matrix from angles in spherical space

    :param theta ```Tensor(..., 1) | float```: rotation angles around y axis
    :param phi  ```Tensor(..., 1) | float```: rotation angles around x axis
    :return: ```Tensor(..., 3, 3)``` rotation matrices
    """
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor([theta])
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor([phi])
    spher = torch.cat([torch.ones_like(theta), theta, phi], dim=-1)
    print(spher)
    forward = SphericalToCartesian(spher)  # (..., 3)
    up = torch.tensor([0.0, 1.0, 0.0])
    forward, up = torch.broadcast_tensors(forward, up)
    print(forward, up)
    right = torch.cross(forward, up, dim=-1)  # (..., 3)
    up = torch.cross(right, forward, dim=-1)  # (..., 3)
    print(right, up, forward)
    return torch.stack([right, up, forward], dim=-2)  # (..., 3, 3)


def broadcast_cat(input: torch.Tensor,
                  s: Union[Number, List[Number], torch.Tensor],
                  dim=-1,
                  append: bool = True) -> torch.Tensor:
    """
    Concatenate a tensor with a scalar along last dimension

    :param input ```Tensor(..., N)```: input tensor
    :param s: scalar
    :param append: append or prepend the scalar to input tensor
    :return: ```Tensor(..., N+1)```
    """
    if dim != -1:
        raise NotImplementedError('currently only support the last dimension')
    if isinstance(s, torch.Tensor):
        x = s
    elif isinstance(s, list):
        x = torch.tensor(s, dtype=input.dtype, device=input.device)
    else:
        x = torch.tensor([s], dtype=input.dtype, device=input.device)
    expand_shape = list(input.size())
    expand_shape[dim] = -1
    x = x.expand(expand_shape)
    return torch.cat([input, x] if append else [x, input], dim)


def generate_video(frames: torch.Tensor, path: str, fps: float,
                   repeat: int = 1, pingpong: bool = False,
                   video_codec: str = 'libx264'):
    """
    Generate video from a sequence of frames after converting type and
    permuting channels to meet the requirement of  ```torchvision.io.write_video()```

    :param frames ```Tensor(B, C, H, W)```: a sequence of frames
    :param path: video path
    :param fps: frames per second
    :param repeat: repeat times
    :param pingpong: whether repeat sequence in pinpong form
    :param video_codec: video codec
    """
    frames = trans_func.convert_image_dtype(frames, torch.uint8)
    frames = frames.detach().cpu().permute(0, 2, 3, 1)
    if pingpong:
        frames = torch.cat([frames, frames.flip(0)], 0)
    frames = frames.expand(repeat, -1, -1, -1, 3).flatten(0, 1)
    torchvision.io.write_video(path, frames, fps, video_codec)
