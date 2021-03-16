from typing import Tuple
import math
import torch
import torch.nn as nn
from my import net_modules
from my import util
from my import device
from my import color_mode
from .msl_net_new import NewMslNet


class Sampler(nn.Module):

    def __init__(self, net: NewMslNet):
        super().__init__()
        self.net = net

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        coords, pts, depths = self.net.sampler(rays_o, rays_d)
        return self.net.input_encoder(coords), depths


class FcNet1(nn.Module):

    def __init__(self, net: NewMslNet):
        super().__init__()
        self.net = net

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.net.nets[0](encoded[:, :self.net.n_samples // 2]),


class FcNet2(nn.Module):

    def __init__(self, net: NewMslNet):
        super().__init__()
        self.net = net

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.net.nets[1](encoded[:, self.net.n_samples // 2:])


class CatNet(nn.Module):

    def __init__(self, net: NewMslNet):
        super().__init__()
        self.net = net

    def forward(self, raw1: torch.Tensor, raw2: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        raw = torch.cat([raw1, raw2], 1)
        colors, alphas = self.net.rendering.raw2color(raw, depths)
        return torch.cat([colors, alphas[..., None]], -1)