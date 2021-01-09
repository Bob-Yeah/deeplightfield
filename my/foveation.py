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

    def to(self, device):
        self.eye_fovea_blend = [x.to(device=device) for x in self.eye_fovea_blend]
        return self

    def synthesis(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Generate foveated retinal image by blending fovea layers
        **Note: current implementation only support two fovea layers**

        :param layers ```List(Tensor(B, C, H'{l}, W'{l}))```: list of foveated layers
        :return ```Tensor(B, C, H:out, W:out)```: foveated images
        """
        output: torch.Tensor = nn_f.interpolate(layers[-1], self.out_res,
                                  mode='bilinear', align_corners=False)
        for i in range(self.n_layers - 2, -1, -1):
            output_roi = output[self.get_layer_region_in_final_image(i)]
            image = nn_f.interpolate(layers[i], output_roi.size()[-2:],
                                     mode='bilinear', align_corners=False)
            blend = self.eye_fovea_blend[i]
            output_roi.mul_(1 - blend).add_(image * blend)
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

    def get_layer_region_in_final_image(self, i: int) -> Tuple[slice, slice]:
        """
        Get region of fovea layer i in final image

        :param i: index of layer
        :return: tuple of slice objects stores the start and end of region in horizontal and vertical
        """
        roi_size = self.get_layer_size_in_final_image(i)
        roi_offset_y = (self.out_res[0] - roi_size) // 2
        roi_offset_x = (self.out_res[1] - roi_size) // 2
        return (...,
            slice(roi_offset_y, roi_offset_y + roi_size),
            slice(roi_offset_x, roi_offset_x + roi_size)
        )

    def _gen_layer_blendmap(self, i: int) -> torch.Tensor:
        """
        Generate blend map for fovea layer i

        :param i: index of fovea layer
        :return ```Tensor(H{i}, W{i})```: blend map
        """
        size = self.get_layer_size_in_final_image(i)
        R = size / 2
        p = util.MeshGrid((size, size)).to(device=self.device)  # (size, size, 2)
        r = torch.norm(p - R, dim=2)  # (size, size, 2)
        return util.SmoothStep(R, R * 0.6, r)
