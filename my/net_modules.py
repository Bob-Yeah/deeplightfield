from typing import List
import torch
import torch.nn as nn
import numpy as np


class FcLayer(nn.Module):

    def __init__(self, in_chns: int, out_chns: int, activate: nn.Module = None,
                 skip_chns: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_chns + skip_chns, out_chns),
            activate
        ) if activate else nn.Linear(in_chns + skip_chns, out_chns)
        self.skip = skip_chns != 0

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x0, x], dim=-1) if self.skip else x)


class FcNet(nn.Module):

    def __init__(self, *, in_chns: int, out_chns: int,
                 nf: int, n_layers: int, skips: List[int] = []):
        """
        Initialize a full-connection net

        :kwarg in_chns: channels of input
        :kwarg out_chns: channels of output
        :kwarg nf: number of features in each hidden layer
        :kwarg n_layers: number of layers
        :kwarg skips: create skip connections from input to layers in this list
        """
        super().__init__()
        self.layers = list()
        self.layers += [FcLayer(in_chns, nf, nn.ReLU())]
        self.layers += [
            FcLayer(nf, nf, nn.ReLU(),
                    skip_chns=in_chns if i in skips else 0)
            for i in range(1, n_layers)
        ]
        self.layers += [FcLayer(nf, out_chns)]
        for i, layer in enumerate(self.layers):
            self.add_module('layer%d' % i, layer)
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer in self.layers:
            x = layer(x, x0)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)


class InputEncoder(nn.Module):

    def Get(multires, input_dims):
        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        return InputEncoder(**embed_kwargs)

    def __init__(self, **kwargs):
        super().__init__()
        self._CreateFunc(**kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encode the given input to R^D space

        :param input ```Tensor(B x C)```: input
        :return ```Tensor(B x D): encoded
        :rtype: torch.Tensor
        """
        return torch.cat([fn(input) for fn in self.embed_fns], dim=-1)

    def _CreateFunc(self, **kwargs):
        embed_fns = []
        d = kwargs['input_dims']
        out_dim = 0

        if kwargs['include_input'] or kwargs['num_freqs'] == 0:
            embed_fns.append(lambda x: x)
            out_dim += d

        if kwargs['num_freqs'] != 0:
            max_freq = kwargs['max_freq_log2']
            N_freqs = kwargs['num_freqs']

            if kwargs['log_sampling']:
                freq_bands = 2. ** np.linspace(0., max_freq, N_freqs)
            else:
                freq_bands = np.linspace(2. ** 0., 2. ** max_freq, N_freqs)

            for freq in freq_bands:
                for p_fn in kwargs['periodic_fns']:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                     freq=freq: p_fn(x * freq))
                    out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
