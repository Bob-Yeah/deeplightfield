import torch
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self, num_channels, upscale_factor, d=64, s=12, m=4):
        super(Net, self).__init__()

        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=d,
                      kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        self.layers = []
        self.layers += [
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        ]
        for _ in range(m):
            self.layers += [
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1),
                nn.PReLU()
            ]
        self.layers += [
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        ]

        self.mid_part = nn.Sequential(*self.layers)

        # Deconvolution
        if upscale_factor % 2:
            self.last_part = nn.ConvTranspose2d(
                in_channels=d, out_channels=num_channels, kernel_size=9,
                stride=upscale_factor, padding=5 - (upscale_factor + 1) // 2)
        else:
            self.last_part = nn.ConvTranspose2d(
                in_channels=d, out_channels=num_channels, kernel_size=9,
                stride=upscale_factor, padding=5 - upscale_factor // 2,
                output_padding=1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()
