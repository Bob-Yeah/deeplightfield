import os
import torch
import glm
import csv
import numpy as np
from typing import List, Mapping, Tuple, Union
from torch.types import Number
from .constants import *


gvec_type = [glm.dvec1, glm.dvec2, glm.dvec3, glm.dvec4]
gmat_type = [[glm.dmat2, glm.dmat2x3, glm.dmat2x4],
             [glm.dmat3x2, glm.dmat3, glm.dmat3x4],
             [glm.dmat4x2, glm.dmat4x3, glm.dmat4]]


def smooth_step(x0, x1, x):
    y = torch.clamp((x - x0) / (x1 - x0), 0, 1)
    return y * y * (3 - 2 * y)


def torch2np(input: torch.Tensor) -> np.ndarray:
    return input.cpu().detach().numpy()


def torch2glm(input):
    input = input.squeeze()
    size = input.size()
    if len(size) == 1:
        if size[0] <= 0 or size[0] > 4:
            raise ValueError
        return gvec_type[size[0] - 1](torch2np(input))
    if len(size) == 2:
        if size[0] <= 1 or size[0] > 4 or size[1] <= 1 or size[1] > 4:
            raise ValueError
        return gmat_type[size[1] - 2][size[0] - 2](torch2np(input))
    raise ValueError


def glm2torch(val) -> torch.Tensor:
    return torch.from_numpy(np.array(val))


def meshgrid(*size: int, normalize: bool = False, swap_dim: bool = False) -> torch.Tensor:
    """
    Generate a mesh grid

    :param *size: grid size (rows, columns)
    :param normalize: return coords in normalized space? defaults to False
    :param swap_dim: if True, return coords in (y, x) order, defaults to False
    :return: rows x columns x 2 tensor
    """
    if len(size) == 1:
        size = (size[0], size[0])
    y, x = torch.meshgrid(torch.arange(0, size[0]), torch.arange(0, size[1]))
    if swap_dim:
        return torch.stack([y / (size[0] - 1.), x / (size[1] - 1.)], 2) if normalize else torch.stack([y, x], 2)
    return torch.stack([x / (size[1] - 1.), y / (size[0] - 1.)], 2) if normalize else torch.stack([x, y], 2)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    angle = -torch.atan(x / y) + (y < 0) * PI + 0.5 * PI
    return angle


def depth_sample(depth_range: Tuple[float, float], n: int, lindisp: bool) -> torch.Tensor:
    """
    Get [n_layers] foreground layers whose diopters are distributed uniformly
    in  [depth_range] plus a background layer

    :param depth_range: depth range of foreground layers
    :param n_layers: number of foreground layers
    :return: list of [n_layers+1] depths
    """
    if lindisp:
        depth_range = (1 / depth_range[0], 1 / depth_range[1])
    samples = torch.linspace(depth_range[0], depth_range[1], n)
    return samples


def broadcast_cat(input: torch.Tensor,
                  s: Union[Number, List[Number], torch.Tensor],
                  dim=-1,
                  append: bool = True) -> torch.Tensor:
    """
    Concatenate a tensor with a scalar along last dimension

    :param input `Tensor(..., N)`: input tensor
    :param s: scalar
    :param append: append or prepend the scalar to input tensor
    :return: `Tensor(..., N+1)`
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


def save_2d_tensor(path, x):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for i in range(x.shape[0]):
            csv_writer.writerow(x[i])


def view_like(input: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Reshape input to be the same size as ref except the last dimension

    :param input `Tensor(..., C)`: input tensor
    :param ref `Tensor(B.., *): reference tensor
    :return `Tensor(B.., C)`: reshaped tensor
    """
    out_shape = list(ref.size())
    out_shape[-1] = -1
    return input.view(out_shape)


def values(map, *keys): return (map[key] for key in keys)
