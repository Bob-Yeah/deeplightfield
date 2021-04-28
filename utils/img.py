import os
import sys
import shutil
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as trans_f
from . import misc
from .constants import TINY_FLOAT


def is_image_file(filename):
    """
    Chech if `filename` is an image file (with extension of .png, .jpg or .jpeg)

    :param filename `str`: name of the file to check
    :return `bool`: whether `filename` is an image file or not
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def np2torch(img, permute=True):
    """
    Convert numpy-images(s) to torch-image(s), permute channels dim if `permute=True`

    :param input `ndarray([B]HWC)`: 3D or 4D numpy-image(s)
    :return `Tensor([B]HWC|[B]CHW)`: 3D or 4D torch-image(s)
    """
    batch_input = len(img.shape) == 4
    if permute:
        t = torch.from_numpy(np.transpose(
            img, [0, 3, 1, 2] if batch_input else [2, 0, 1]))
    else:
        t = torch.from_numpy(img)
    if not batch_input:
        t = t.unsqueeze(0)
    return t


def torch2np(input: torch.Tensor) -> np.ndarray:
    """
    Convert torch-image(s) to numpy-images(s) with channels at the last dim

    :param input `Tensor(HW|[B]CHW|[B]HWC)`: 2D, 3D or 4D torch-image(s)
    :return `ndarray ([B]HWC)`: numpy-image(s) with channels transposed to the last dim
    """
    img = misc.torch2np(input.squeeze())
    if len(img.shape) == 2:  # 2D(HW): Single channel image
        return img
    batch_input = len(img.shape) == 4
    if input.size()[batch_input] <= 4:  # 3D(CHW) or 4D(BCHW): transpose channel
        return np.transpose(img, [0, 2, 3, 1] if batch_input else [1, 2, 0])
    return img


def load(*paths: str, permute=True, with_alpha=False) -> torch.Tensor:
    """
    Load one or multiple torch-image(s) from `paths`

    :param *paths `str...`: path of image(s) to load
    :param permute `bool`: whether permute channels dim, defaults to `True`
    :param with_alpha `bool`:whether load with alpha channel , defaults to `False`, which means only RGB channels are loaded
    :return `Tensor(BCHW|BHWC)`: loaded torch-image(s)
    """
    chns = 4 if with_alpha else 3
    new_paths = []
    for path in paths:
        new_paths += [path] if isinstance(path, str) else list(path)
    imgs = np.stack([plt.imread(path)[..., :chns] for path in new_paths])
    if imgs.dtype == 'uint8':
        imgs = imgs.astype(np.float32) / 255
    return np2torch(imgs, permute)


def load_seq(path: str, n: int, permute=True, with_alpha=False) -> torch.Tensor:
    return load([path % i for i in range(n)], permute=permute, with_alpha=with_alpha)


def save(input: torch.Tensor, *paths: str):
    """
    Save one or multiple torch-image(s) to `paths`

    :param input `torch.Tensor`: torch-image(s) to save
    :param *paths `str...`: paths to save torch-image(s) to
    :raises `ValueError`: if number of paths does not match batches of input image(s)
    """
    new_paths = []
    for path in paths:
        new_paths += [path] if isinstance(path, str) else list(path)
    if (len(input.size()) != 4 and len(new_paths) != 1) or input.size(0) != len(new_paths):
        raise ValueError
    np_img = torch2np(input)
    if np_img.dtype.kind == 'f':
        np_img = np.clip(np_img, 0, 1)
    if not np_img.flags['C_CONTIGUOUS']:
        np_img = np.ascontiguousarray(np_img)
    for i, path in enumerate(new_paths):
        plt.imsave(path, np_img[i])


def save_seq(input: torch.Tensor, path: str):
    n = 1 if len(input.size()) <= 3 else input.size(0)
    return save(input, [path % i for i in range(n)])


def plot(input: torch.Tensor, *, ax: plt.Axes = None):
    """
    Plot a torch-image using matplotlib

    :param input `Tensor(HW|[B]CHW|[B]HWC)`: 2D, 3D or 4D torch-image(s)
    :param ax `plt.Axes`: (optional) specify the axes to plot image
    """
    return plt.imshow(torch2np(input)) if ax is None else ax.imshow(torch2np(input))


def save_video(frames: torch.Tensor, path: str, fps: int,
               repeat: int = 1, pingpong: bool = False):
    """
    Encode and save a sequence of frames as video file

    :param frames `Tensor(B, C, H, W)`: a sequence of frames
    :param path: video path
    :param fps: frames per second
    :param repeat: repeat times
    :param pingpong: whether repeat sequence in pinpong form
    :param video_codec: video codec
    """
    if pingpong:
        frames = torch.cat([frames, frames.flip(0)], 0)
    if repeat > 1:
        frames = frames.expand(repeat, -1, -1, -1, -1).flatten(0, 1)
    dir, file_name = os.path.split(path)
    misc.create_dir(dir)
    cwd = os.getcwd()
    os.chdir(dir)
    temp_out_dir = os.path.splitext(file_name)[0] + '_tempout'
    misc.create_dir(temp_out_dir)
    os.chdir(temp_out_dir)
    save_seq(frames, 'out_%04d.png')
    os.system(f'ffmpeg -y -r {fps:d} -i out_%04d.png -c:v libx264 -vf fps={fps:d} -pix_fmt yuv420p ../{file_name}')
    os.chdir(cwd)
    shutil.rmtree(os.path.join(dir, temp_out_dir))


def horizontal_shift(input: torch.Tensor, offset: int, dim=-1) -> torch.Tensor:
    if offset == 0:
        return input
    shifted = torch.zeros_like(input)
    if dim == -1:
        if offset > 0:
            shifted[..., offset:] = input[..., :-offset]
        else:
            shifted[..., :offset] = input[..., -offset:]
    elif dim == -2:
        if offset > 0:
            shifted[..., offset:, :] = input[..., :-offset, :]
        else:
            shifted[..., :offset, :] = input[..., -offset:, :]
    else:
        raise NotImplementedError
    return shifted


def mse2psnr(x):
    logfunc = torch.log if isinstance(x, torch.Tensor) else np.log
    return -10. * logfunc(x + TINY_FLOAT) / np.log(10.)


def colorize_depthmap(depthmap: torch.Tensor, depth_range, inverse=True, colormap='binary'):
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if inverse:
        depthmap = torch.reciprocal(depthmap)
        depth_range = (1 / depth_range[0], 1 / depth_range[1])
    depthmap = (depthmap - depth_range[0]) / (depth_range[1] - depth_range[0])
    depthmap = misc.torch2np(depthmap)
    return torch.tensor(colormap(depthmap))