import torch
from typing import List
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, loss_modules: List[nn.Module], weights: List[float]):
        super().__init__()
        self.loss_modules = nn.ModuleList(loss_modules)
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum([self.weights[i] * self.loss_modules[i](input, target)
                    for i in range(len(self.loss_modules))])


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_dy, input_dx = self._cal_grad(input)
        target_dy, target_dx = self._cal_grad(target)
        return self.mse_loss(
            torch.cat([
                input_dy.flatten(1, -1),
                input_dx.flatten(1, -1)
            ], 1),
            torch.cat([
                target_dy.flatten(1, -1),
                target_dx.flatten(1, -1)
            ], 1))

    def _cal_grad(self, images):
        """
        Calculate gradient of images

        :param image `Tensor(..., C, H, W)`: input images
        :return `Tensor(..., 2C, H-2, W-2)`: gradient map of input images
        """
        dy = images[..., 2:, :] - images[..., :-2, :]
        dx = images[..., :, 2:] - images[..., :, :-2]
        return dy, dx
