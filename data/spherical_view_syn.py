import os
import json
import torch
import glm
import torch.nn.functional as nn_f
from typing import Tuple, Union
from utils import img
from utils import device
from utils import view
from utils import color


class SphericalViewSynDataset(object):
    """
    Data loader for spherical view synthesis task

    Attributes
    --------
    data_dir ```str```: the directory of dataset\n
    view_file_pattern ```str```: the filename pattern of view images\n
    cam_params ```object```: camera intrinsic parameters\n
    view_centers ```Tensor(N, 3)```: centers of views\n
    view_rots ```Tensor(N, 3, 3)```: rotation matrices of views\n
    view_images ```Tensor(N, 3, H, W)```: images of views\n
    view_depths ```Tensor(N, H, W)```: depths of views\n
    """

    def __init__(self, dataset_desc_path: str, load_images: bool = True,
                 load_depths: bool = False, load_bins: bool = False, c: int = color.RGB,
                 calculate_rays: bool = True, res: Tuple[int, int] = None):
        """
        Initialize data loader for spherical view synthesis task

        The dataset description file is a JSON file with following fields:

        - view_file_pattern: string, the path pattern of view images
        - view_res: { "x", "y" }, the resolution of view images
        - cam_params: { "fx", "fy", "cx", "cy" }, the focal and center of camera (in normalized image space)
        - view_centers: [ [ x, y, z ], ... ], centers of views
        - view_rots: [ [ m00, m01, ..., m22 ], ... ], rotation matrices of views

        :param dataset_desc_path ```str```: path to the data description file
        :param load_images ```bool```: whether load view images and return in __getitem__()
        :param load_depths ```bool```: whether load depth images and return in __getitem__()
        :param c ```int```: color space to convert view images to
        :param calculate_rays ```bool```: whether calculate rays
        """
        super().__init__()
        self.data_dir = os.path.dirname(dataset_desc_path)
        self.load_images = load_images
        self.load_depths = load_depths
        self.load_bins = load_bins

        # Load dataset description file
        self._load_desc(dataset_desc_path, res)

        # Load view images
        if self.load_images:
            self.view_images = color.cvt(
                img.load(self.view_file % i for i in self.view_idxs).to(device.default()),
                color.RGB, c)
            if res:
                self.view_images = nn_f.interpolate(self.view_images, res)
        else:
            self.view_images = None

        # Load depthmaps
        if self.load_depths:
            self.view_depths = self._decode_depth_images(
                img.load(self.depth_file % i for i in self.view_idxs).to(device.default()))
            if res:
                self.view_depths = nn_f.interpolate(self.view_depths, res)
        else:
            self.view_depths = None

        # Load depthmaps
        if self.load_bins:
            self.view_bins = img.load([self.bins_file % i for i in self.view_idxs], permute=False) \
                .to(device.default())
            if res:
                self.view_bins = nn_f.interpolate(self.view_bins, res)
        else:
            self.view_bins = None

        self.patched_images = self.view_images
        self.patched_depths = self.view_depths
        self.patched_bins = self.view_bins

        if calculate_rays:
            # rays_o & rays_d are both (N, H, W, 3)
            self.rays_o, self.rays_d = self.cam_params.get_global_rays(
                view.Trans(self.view_centers, self.view_rots))
            self.patched_rays_o = self.rays_o
            self.patched_rays_d = self.rays_d

    def _decode_depth_images(self, input):
        disp_range = (1 / self.depth_range[0], 1 / self.depth_range[1])
        disp_val = (1 - input[..., 0, :, :]) * (disp_range[1] - disp_range[0]) + disp_range[0]
        return torch.reciprocal(disp_val)

    def _euler_to_matrix(self, euler):
        q = glm.quat(glm.radians(glm.vec3(euler[0], euler[1], euler[2])))
        return glm.transpose(glm.mat3_cast(q)).to_list()

    def _load_desc(self, path, res=None):
        with open(path, 'r', encoding='utf-8') as file:
            data_desc = json.loads(file.read())
        if not data_desc.get('view_file_pattern'):
            self.load_images = False
        else:
            self.view_file = os.path.join(self.data_dir, data_desc['view_file_pattern'])
        if not data_desc.get('depth_file_pattern'):
            self.load_depths = False
        else:
            self.depth_file = os.path.join(self.data_dir, data_desc['depth_file_pattern'])
        if not data_desc.get('bins_file_pattern'):
            self.load_bins = False
        else:
            self.bins_file = os.path.join(self.data_dir, data_desc['bins_file_pattern'])
        self.view_res = res if res else (data_desc['view_res']['y'], data_desc['view_res']['x'])
        self.cam_params = view.CameraParam(data_desc['cam_params'], self.view_res,
                                           device=device.default())
        self.depth_range = [data_desc['depth_range']['min'], data_desc['depth_range']['max']] \
            if 'depth_range' in data_desc else None
        self.range = [data_desc['range']['min'], data_desc['range']['max']] \
            if 'range' in data_desc else None
        self.samples = data_desc['samples'] if 'samples' in data_desc else None
        self.view_centers = torch.tensor(
            data_desc['view_centers'], device=device.default())  # (N, 3)
        self.view_rots = torch.tensor(
            [self._euler_to_matrix([rot[1], rot[0], 0]) for rot in data_desc['view_rots']]
            if len(data_desc['view_rots'][0]) == 2 else data_desc['view_rots'],
            device=device.default()).view(-1, 3, 3)  # (N, 3, 3)
        #self.view_centers = self.view_centers[:6]
        #self.view_rots = self.view_rots[:6]
        self.n_views = self.view_centers.size(0)
        self.n_pixels = self.n_views * self.view_res[0] * self.view_res[1]
        self.view_idxs = data_desc['views'][:self.n_views] if 'views' in data_desc else range(self.n_views)

        if 'gl_coord' in data_desc and data_desc['gl_coord'] == True:
            print('Convert from OGL coordinate to DX coordinate (i. e. flip z axis)')
            if not data_desc['cam_params'].get('normalized'):
                self.cam_params.f[1] *= -1
            self.view_centers[:, 2] *= -1
            self.view_rots[:, 2] *= -1
            self.view_rots[..., 2] *= -1

    def set_patch_size(self, patch_size: Union[int, Tuple[int, int]],
                       offset: Union[int, Tuple[int, int]] = 0):
        """
        Set the size of patch and (optional) offset. If patch_size = (1, 1)

        :param patch_size: 
        :param offset: 
        """
        if not isinstance(patch_size, tuple):
            patch_size = (int(patch_size), int(patch_size))
        if not isinstance(offset, tuple):
            offset = (int(offset), int(offset))
        patches = ((self.view_res[0] - offset[0]) // patch_size[0],
                   (self.view_res[1] - offset[1]) // patch_size[1])
        slices = (..., slice(offset[0], offset[0] + patches[0] * patch_size[0]),
                  slice(offset[1], offset[1] + patches[1] * patch_size[1]))
        ray_slices = (slice(self.n_views),
                      slice(offset[0], offset[0] + patches[0] * patch_size[0]),
                      slice(offset[1], offset[1] + patches[1] * patch_size[1]))
        if patch_size[0] == 1 and patch_size[1] == 1:
            self.patched_images = self.view_images[slices] \
                .permute(0, 2, 3, 1).flatten(0, 2) if self.load_images else None
            self.patched_depths = self.view_depths[slices].flatten() if self.load_depths else None
            self.patched_bins = self.view_bins[slices].flatten(0, 2) if self.load_bins else None
            self.patched_rays_o = self.rays_o[ray_slices].flatten(0, 2)
            self.patched_rays_d = self.rays_d[ray_slices].flatten(0, 2)
        elif patch_size[0] == self.view_res[0] and patch_size[1] == self.view_res[1]:
            self.patched_images = self.view_images
            self.patched_depths = self.view_depths
            self.patched_bins = self.view_bins
            self.patched_rays_o = self.rays_o
            self.patched_rays_d = self.rays_d
        else:
            self.patched_images = self.view_images[slices] \
                .view(self.n_views, -1, patches[0], patch_size[0], patches[1], patch_size[1]) \
                .permute(0, 2, 4, 1, 3, 5).flatten(0, 2) if self.load_images else None
            self.patched_depths = self.view_depths[slices] \
                .view(self.n_views, patches[0], patch_size[0], patches[1], patch_size[1]) \
                .permute(0, 1, 3, 2, 4).flatten(0, 2) if self.load_depths else None
            self.patched_bins = self.view_bins[slices] \
                .view(self.n_views, patches[0], patch_size[0], patches[1], patch_size[1], -1) \
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 2) if self.load_bins else None
            self.patched_rays_o = self.rays_o[ray_slices] \
                .view(self.n_views, patches[0], patch_size[0], patches[1], patch_size[1], -1) \
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 2)
            self.patched_rays_d = self.rays_d[ray_slices] \
                .view(self.n_views, patches[0], patch_size[0], patches[1], patch_size[1], -1) \
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 2)

    def __len__(self):
        return self.patched_rays_o.size(0)

    def __getitem__(self, idx):
        return idx, self.patched_images[idx] if self.load_images else None, \
            self.patched_rays_o[idx], self.patched_rays_d[idx]
