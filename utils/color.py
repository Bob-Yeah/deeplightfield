from typing import Union
import torch
import torchvision.transforms.functional as vtf


RGB = 0
GRAY = 1
YCbCr = 2


def to_str(color):
    return "gray" if color == GRAY \
        else ("ybr" if color == YCbCr
              else "rgb")


def from_str(color_str):
    return GRAY if color_str == 'gray' \
        else (YCbCr if color_str == 'ybr'
              else RGB)


def chns(color):
    color = from_str(color) if isinstance(color, str) else color
    return 1 if color == GRAY else 3


def noop(input: torch.Tensor) -> torch.Tensor:
    return input


def rgb2ycbcr(input: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor from RGB to YCbCr

    :param input `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    :return `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    """
    if input.size(-1) == 3:
        r = input[..., 0:1]
        g = input[..., 1:2]
        b = input[..., 2:3]
        dim_c = -1
    else:
        r = input[..., 0:1, :, :]
        g = input[..., 1:2, :, :]
        b = input[..., 2:3, :, :]
        dim_c = -3
    y = r * 0.25678824 + g * 0.50412941 + b * 0.09790588 + 0.0625
    cb = r * -0.14822353 + g * -0.29099216 + b * 0.43921569 + 0.5
    cr = r * 0.43921569 + g * -0.36778824 + b * -0.07142745 + 0.5
    return torch.cat([y, cb, cr], dim_c)


def rgb2ycbcr(input: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor from RGB to YCbCr

    :param input `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    :return `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    """
    if input.size(-1) == 3:
        r = input[..., 0:1]
        g = input[..., 1:2]
        b = input[..., 2:3]
        dim_c = -1
    else:
        r = input[..., 0:1, :, :]
        g = input[..., 1:2, :, :]
        b = input[..., 2:3, :, :]
        dim_c = -3
    y = r * 0.257 + g * 0.504 + b * 0.098 + 0.0625
    cb = r * -0.148 + g * -0.291 + b * 0.439 + 0.5
    cr = r * 0.439 + g * -0.368 + b * -0.071 + 0.5
    return torch.cat([y, cb, cr], dim_c)


def ycbcr2rgb(input: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor from YCbCr to RGB

    :param input `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    :return `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    """
    if input.size(-1) == 3:
        y = input[..., 0:1]
        cb = input[..., 1:2]
        cr = input[..., 2:3]
        dim_c = -1
    else:
        y = input[..., 0:1, :, :]
        cb = input[..., 1:2, :, :]
        cr = input[..., 2:3, :, :]
        dim_c = -3
    y = y - 0.0625
    cb = cb - 0.5
    cr = cr - 0.5
    r = y * 1.164 + cr * 1.596
    g = y * 1.164 + cb * -0.392 + cr * -0.813
    b = y * 1.164 + cb * 2.017
    return torch.cat([r, g, b], dim_c)


def gray2rgb(input: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor from GRAY to RGB

    :param input `Tensor(..., 1) | Tensor(..., 1, H, W)`: 
    :return `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    """
    out_size = list(input.size())
    if input.size(-1) == 1:
        out_size[-1] = 3
    else:
        out_size[-3] = 3
    return input.expand(out_size)


def gray2ycbcr(input: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor from GRAY to YCbCr

    :param input `Tensor(..., 1) | Tensor(..., 1, H, W)`: 
    :return `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    """
    if input.size(-1) == 3:
        dim_c = -1
    else:
        dim_c = -3
    y = input * 0.859 + 0.0625
    cb = cr = torch.ones_like(input) * 0.5
    return torch.cat([y, cb, cr], dim_c)


def ycbcr2gray(input: torch.Tensor) -> torch.Tensor:
    """
    Convert input tensor from YCbCr to GRAY

    :param input `Tensor(..., 3) | Tensor(..., 3, H, W)`: 
    :return `Tensor(..., 1) | Tensor(..., 1, H, W)`: 
    """
    return vtf.rgb_to_grayscale(ycbcr2rgb(input))


cvt_funcs = [
    [noop, vtf.rgb_to_grayscale, rgb2ycbcr],  # RGB->RGB,GRAY,YCbCr
    [gray2rgb, noop, gray2ycbcr],  # GRAY->RGB,GRAY,YCbCr
    [ycbcr2rgb, ycbcr2gray, noop]  # YCbCr->RGB,GRAY,YCbCr
]


def cvt(input: torch.Tensor, from_color: Union[int, str], to_color: Union[int, str]) -> torch.Tensor:
    """
    Convert input tensor from `from_color` to `to_color`

    :param input `Tensor(..., 1|3) | Tensor(..., 1|3, H, W)`: The image to convert
    :param from_color `int | str`: The color of input image
    :param to_color `int | str`: The color of output image
    :return `Tensor(..., 1|3) | Tensor(..., 1|3, H, W)`: converted image
    """
    if isinstance(from_color, str):
        from_color = from_str(from_color)
    if isinstance(to_color, str):
        to_color = from_str(to_color)
    return cvt_funcs[from_color][to_color](input)
