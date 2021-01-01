import math
from typing import Tuple
import torch
import torchvision.transforms.functional as trans_f
import json
from ..my import util
from ..my import device


def _convert_camera_params(input_camera_params, view_res):
    """
    Check and convert camera parameters in config file to pixel-space

    :param cam_params: { ["fx", "fy" | "fov"], "cx", "cy", ["normalized"] },
        the parameters of camera
    :return: camera parameters
    """
    input_is_normalized = bool(input_camera_params.get('normalized'))
    camera_params = {}
    if 'fov' in input_camera_params:
        camera_params['fx'] = camera_params['fy'] = \
            (1 if input_is_normalized else view_res[0]) / \
            util.Fov2Length(input_camera_params['fov'])
        camera_params['fy'] *= -1
    else:
        camera_params['fx'] = input_camera_params['fx']
        camera_params['fy'] = input_camera_params['fy']
    camera_params['cx'] = input_camera_params['cx']
    camera_params['cy'] = input_camera_params['cy']
    if input_is_normalized:
        camera_params['fx'] *= view_res[1]
        camera_params['fy'] *= view_res[0]
        camera_params['cx'] *= view_res[1]
        camera_params['cy'] *= view_res[0]
    return camera_params


class SphericalViewSynDataset(torch.utils.data.dataset.Dataset):
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
    """

    def __init__(self, dataset_desc_path: str, load_images: bool = True, gray: bool = False,
                 ray_as_item=False):
        """
        Initialize data loader for spherical view synthesis task

        The dataset description file is a JSON file with following fields:

        - view_file_pattern: string, the path pattern of view images
        - view_res: { "x", "y" }, the resolution of view images
        - cam_params: { ["fx", "fy" | "fov"], "cx", "cy", ["normalized"] }, the parameters of camera
        - view_centers: [ [ x, y, z ], ... ], centers of views
        - view_rots: [ [ m00, m01, ..., m22 ], ... ], rotation matrices of views

        :param dataset_desc_path ```str```: path to the data description file
        :param load_images ```bool```: whether load view images and return in __getitem__()
        :param gray ```bool```: whether convert view images to grayscale
        :param ray_as_item ```bool```: whether to treat each ray in view as an item
        """
        self.data_dir = dataset_desc_path.rsplit('/', 1)[0] + '/'
        self.load_images = load_images
        self.ray_as_item = ray_as_item

        # Load dataset description file
        with open(dataset_desc_path, 'r', encoding='utf-8') as file:
            data_desc = json.loads(file.read())
        if data_desc['view_file_pattern'] == '':
            self.load_images = False
        else:
            self.view_file_pattern: str = self.data_dir + \
                data_desc['view_file_pattern']
        self.view_res = (data_desc['view_res']['y'],
                         data_desc['view_res']['x'])
        self.cam_params = _convert_camera_params(
            data_desc['cam_params'], self.view_res)
        self.view_centers = torch.tensor(data_desc['view_centers'])  # (N, 3)
        self.view_rots = torch.tensor(data_desc['view_rots']) \
            .view(-1, 3, 3)  # (N, 3, 3)

        # Load view images
        if self.load_images:
            self.view_images = util.ReadImageTensor(
                [self.view_file_pattern % i for i in range(self.view_centers.size(0))])
            if gray:
                self.view_images = trans_f.rgb_to_grayscale(self.view_images)
        else:
            self.view_images = None

        local_view_rays = util.GetLocalViewRays(self.cam_params,
                                                self.view_res,
                                                flatten=True)  # (M, 3)
        # Transpose matrix so we can perform vec x mat
        view_rots_t = self.view_rots.permute(0, 2, 1)

        # rays_o & rays_d are both (N, M, 3)
        self.rays_o = self.view_centers.unsqueeze(1) \
            .expand(-1, local_view_rays.size(0), -1)
        self.rays_d = torch.matmul(local_view_rays, view_rots_t)

        # Flatten rays if ray_as_item = True
        if ray_as_item:
            self.view_pixels = self.view_images.permute(0, 2, 3, 1).flatten(
                0, 2) if self.view_images != None else None
            self.rays_o = self.rays_o.flatten(0, 1)
            self.rays_d = self.rays_d.flatten(0, 1)

    def __len__(self):
        return self.rays_o.size(0)

    def __getitem__(self, idx):
        if self.load_images:
            if self.ray_as_item:
                return idx, self.view_pixels[idx], self.rays_o[idx], self.rays_d[idx]
            return idx, self.view_images[idx], self.rays_o[idx], self.rays_d[idx]
        return idx, False, self.rays_o[idx], self.rays_d[idx]


class FastSphericalViewSynDataset(torch.utils.data.dataset.Dataset):
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
    """

    def __init__(self, dataset_desc_path: str, load_images: bool = True, gray: bool = False):
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
        :param gray ```bool```: whether convert view images to grayscale
        """
        super().__init__()
        self.data_dir = dataset_desc_path.rsplit('/', 1)[0] + '/'
        self.load_images = load_images

        # Load dataset description file
        with open(dataset_desc_path, 'r', encoding='utf-8') as file:
            data_desc = json.loads(file.read())
        if data_desc['view_file_pattern'] == '':
            self.load_images = False
        else:
            self.view_file_pattern: str = self.data_dir + \
                data_desc['view_file_pattern']
        self.view_res = (data_desc['view_res']['y'],
                         data_desc['view_res']['x'])
        self.cam_params = _convert_camera_params(
            data_desc['cam_params'], self.view_res)
        self.view_centers = torch.tensor(
            data_desc['view_centers'], device=device.GetDevice())  # (N, 3)
        self.view_rots = torch.tensor(
            data_desc['view_rots'], device=device.GetDevice()).view(-1, 3, 3)  # (N, 3, 3)
        self.n_views = self.view_centers.size(0)
        self.n_pixels = self.n_views * self.view_res[0] * self.view_res[1]

        # Load view images
        if self.load_images:
            self.view_images = util.ReadImageTensor(
                [self.view_file_pattern % i
                 for i in range(self.view_centers.size(0))]
            ).to(device.GetDevice())
            if gray:
                self.view_images = trans_f.rgb_to_grayscale(self.view_images)
        else:
            self.view_images = None

        local_view_rays = util.GetLocalViewRays(self.cam_params, self.view_res, True) \
            .to(device.GetDevice())  # (HW, 3)
        # Transpose matrix so we can perform vec x mat
        view_rots_t = self.view_rots.permute(0, 2, 1)

        # rays_o & rays_d are both (N, H, W, 3)
        self.rays_o = self.view_centers[:, None, None, :] \
            .expand(-1, self.view_res[0], self.view_res[1], -1)
        self.rays_d = torch.matmul(local_view_rays, view_rots_t) \
            .view_as(self.rays_o)
        self.patched_images = self.view_images  # (N, 1|3, H, W)
        self.patched_rays_o = self.rays_o  # (N, H, W, 3)
        self.patched_rays_d = self.rays_d  # (N, H, W, 3)

    def set_patch_size(self, patch_size: Tuple[int, int], offset: Tuple[int, int] = (0, 0)):
        """
        Set the size of patch and (optional) offset. If patch_size = (1, 1)

        :param patch_size: 
        :param offset: 
        """
        patches = ((self.view_res[0] - offset[0]) // patch_size[0],
                   (self.view_res[1] - offset[1]) // patch_size[1])
        slices = (..., slice(offset[0], offset[0] + patches[0] * patch_size[0]),
                  slice(offset[1], offset[1] + patches[1] * patch_size[1]))
        if patch_size[0] == 1 and patch_size[1] == 1:
            self.patched_images = self.view_images[slices] \
                .permute(0, 2, 3, 1).flatten(0, 2) if self.load_images else None
            self.patched_rays_o = self.rays_o[slices].flatten(0, 2)
            self.patched_rays_d = self.rays_d[slices].flatten(0, 2)
        elif patch_size[0] == self.view_res[0] and patch_size[1] == self.view_res[1]:
            self.patched_images = self.view_images
            self.patched_rays_o = self.rays_o
            self.patched_rays_d = self.rays_d
        else:
            print(self.view_images.size(), self.rays_o.size())
            print(self.view_images[slices].size(), self.rays_o[slices].size())
            self.patched_images = self.view_images[slices] \
                .view(self.n_views, -1, patches[0], patch_size[0], patches[1], patch_size[1]) \
                .permute(0, 2, 4, 1, 3, 5).flatten(0, 2) if self.load_images else None
            self.patched_rays_o = self.rays_o[slices] \
                .view(self.n_views, patches[0], patch_size[0], patches[1], patch_size[1], -1) \
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 2)
            self.patched_rays_d = self.rays_d[slices] \
                .view(self.n_views, patches[0], patch_size[0], patches[1], patch_size[1], -1) \
                .permute(0, 1, 3, 2, 4, 5).flatten(0, 2)

    def __len__(self):
        return self.patched_rays_o.size(0)

    def __getitem__(self, idx):
        if self.load_images:
            return idx, self.patched_images[idx], self.patched_rays_o[idx], \
                self.patched_rays_d[idx]
        return idx, False, self.patched_rays_o[idx], self.patched_rays_d[idx]


class FastDataLoader(object):

    class Iter(object):

        def __init__(self, dataset, batch_size, shuffle, drop_last) -> None:
            super().__init__()
            self.indices = torch.randperm(len(dataset), device=device.GetDevice()) \
                if shuffle else torch.arange(len(dataset), device=device.GetDevice())
            self.offset = 0
            self.batch_size = batch_size
            self.dataset = dataset
            self.drop_last = drop_last

        def __next__(self):
            if self.offset + (self.batch_size if self.drop_last else 0) >= len(self.dataset):
                raise StopIteration()
            indices = self.indices[self.offset:self.offset + self.batch_size]
            self.offset += self.batch_size
            return self.dataset[indices]

    def __init__(self, dataset, batch_size, shuffle, drop_last, **kwargs) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        return FastDataLoader.Iter(self.dataset, self.batch_size,
                                   self.shuffle, self.drop_last)

    def __len__(self):
        return math.floor(len(self.dataset) / self.batch_size) if self.drop_last \
            else math.ceil(len(self.dataset) / self.batch_size)
