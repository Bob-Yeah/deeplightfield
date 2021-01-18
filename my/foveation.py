import math
import torch
import torch.nn.functional as nn_f
from typing import List, Tuple
from . import util


class Foveation(object):

    def __init__(self, fov_list: List[float],
                 out_res: Tuple[int, int], *, device=None):
        self.fov_list = fov_list
        self.out_res = out_res
        self.device = device
        self.n_layers = len(self.fov_list)
        self.eye_fovea_blend = [
            self._gen_layer_blendmap(i)
            for i in range(self.n_layers - 1)
        ]  # blend maps of fovea layers
        self.coords = util.MeshGrid(out_res).to(device=device)

    def to(self, device):
        self.eye_fovea_blend = [x.to(device=device)
                                for x in self.eye_fovea_blend]
        self.coords = self.coords.to(device=device)
        return self

    def synthesis(self, layers: List[torch.Tensor],
                  fovea_center: Tuple[float, float]) -> torch.Tensor:
        """
        Generate foveated retinal image by blending fovea layers
        **Note: current implementation only support two fovea layers**

        :param layers ```List(Tensor(B, C, H'{l}, W'{l}))```: list of foveated layers
        :return ```Tensor(B, C, H:out, W:out)```: foveated images
        """
        output: torch.Tensor = nn_f.interpolate(layers[-1], self.out_res,
                                                mode='bilinear', align_corners=False)
        c = torch.tensor([
            fovea_center[0] + self.out_res[1] / 2,
            fovea_center[1] + self.out_res[0] / 2
        ], device=self.coords.device)
        for i in range(self.n_layers - 2, -1, -1):
            if layers[i] == None:
                continue
            R = self.get_layer_size_in_final_image(i) / 2
            grid = ((self.coords - c) / R)[None, ...]
            blend = nn_f.grid_sample(self.eye_fovea_blend[i][None, None, ...], grid) # (1, 1, H:out, W:out)
            output.mul_(1 - blend).add_(nn_f.grid_sample(layers[i], grid) * blend)
        return output

    def get_layer_size_in_final_image(self, i: int) -> int:
        """
        Get size of layer i in final image

        :param i: index of layer
        :return: size of layer i in final image (in pixels)
        """
        length_i = util.Fov2Length(self.fov_list[i])
        length = util.Fov2Length(self.fov_list[-1])
        k = length_i / length
        return int(math.ceil(self.out_res[0] * k))

    def _gen_layer_blendmap(self, i: int) -> torch.Tensor:
        """
        Generate blend map for fovea layer i

        :param i: index of fovea layer
        :return ```Tensor(H{i}, W{i})```: blend map
        """
        size = self.get_layer_size_in_final_image(i)
        R = size / 2
        p = util.MeshGrid((size, size)).to(
            device=self.device)  # (size, size, 2)
        r = torch.norm(p - R, dim=2)  # (size, size, 2)
        return util.SmoothStep(R, R * 0.6, r)
