import torch
import torchvision.transforms.functional as trans_f
import json
from ..my import util
from ..my import imgio


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
        - cam_params: { "fx", "fy", "cx", "cy" }, the focal and center of camera (in normalized image space)
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
        self.view_file_pattern: str = self.data_dir + \
            data_desc['view_file_pattern']
        self.view_res = (data_desc['view_res']['y'],
                         data_desc['view_res']['x'])
        self.cam_params = data_desc['cam_params']
        self.view_centers = torch.tensor(data_desc['view_centers'])  # (N, 3)
        self.view_rots = torch.tensor(data_desc['view_rots']) \
            .view(-1, 3, 3)  # (N, 3, 3)

        # Load view images
        if load_images:
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

        # ray_positions & ray_directions are both (N, M, 3)
        self.ray_positions = self.view_centers.unsqueeze(1) \
            .expand(-1, local_view_rays.size(0), -1)
        self.ray_directions = torch.matmul(local_view_rays, view_rots_t)

        # Flatten rays if ray_as_item = True
        if ray_as_item:
            self.view_pixels = self.view_images.permute(
                0, 2, 3, 1).flatten(0, 2)
            self.ray_positions = self.ray_positions.flatten(0, 1)
            self.ray_directions = self.ray_directions.flatten(0, 1)

    def __len__(self):
        return self.ray_positions.size(0)

    def __getitem__(self, idx):
        if self.load_images:
            if self.ray_as_item:
                return idx, self.view_pixels[idx], self.ray_positions[idx], self.ray_directions[idx]
            return idx, self.view_images[idx], self.ray_positions[idx], self.ray_directions[idx]
        return idx, self.ray_positions[idx], self.ray_directions[idx]
