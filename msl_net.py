from typing import List
import torch
import torch.nn as nn
from .pytorch_prototyping.pytorch_prototyping import *
from .my import util
from .my import device


class FcNet(nn.Module):

    def __init__(self, in_chns, out_chns, nf, n_layers):
        super().__init__()
        self.layers = list()
        self.layers.append(nn.Linear(in_chns, nf))
        self.layers.append(nn.LeakyReLU())
        for _ in range(1, n_layers):
            self.layers.append(nn.Linear(nf, nf))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(nf, out_chns))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

class Rendering(nn.Module):

    def __init__(self, n_sphere_layers):
        super().__init__()
        self.n_sl = n_sphere_layers

    def forward(self, net, pos, dir):
        """
        [summary]

        :param pos: B x 3, position of a ray
        :param dir: B x 3, direction of a ray
        """
        

class MslNet(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
