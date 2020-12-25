from typing import List, Tuple
import torch
import json
from ..my import util


def ReadLightField(path: str, views: Tuple[int, int], flatten_views: bool = False) -> torch.Tensor:
    input_img = util.ReadImageTensor(path, batch_dim=False)
    h = input_img.size()[1] // views[0]
    w = input_img.size()[2] // views[1]
    if flatten_views:
        lf = torch.empty(views[0] * views[1], 3, h, w)
        for y_i in range(views[0]):
            for x_i in range(views[1]):
                lf[y_i * views[1] + x_i, :, :, :] = \
                    input_img[:, y_i * h:(y_i + 1) * h, x_i * w:(x_i + 1) * w]
    else:
        lf = torch.empty(views[0], views[1], 3, h, w)
        for y_i in range(views[0]):
            for x_i in range(views[1]):
                lf[y_i, x_i, :, :, :] = \
                    input_img[:, y_i * h:(y_i + 1) * h, x_i * w:(x_i + 1) * w]
    return lf


def DecodeDepth(depth_images: torch.Tensor) -> torch.Tensor:
    return depth_images[:, 0].unsqueeze(1).mul(255) / 10


class LightFieldSynDataset(torch.utils.data.dataset.Dataset):
    """
    Data loader for light field synthesis task

    Attributes
    --------
    data_dir ```string```: the directory of dataset\n
    n_views ```tuple(int, int)```: rows and columns of views\n
    num_views ```int```: number of views\n
    view_images ```N x H x W Tensor```: images of views\n
    view_depths ```N x H x W Tensor```: depths of views\n
    view_positions ```N x 3 Tensor```: positions of views\n
    sparse_view_images ```N' x H x W Tensor```: images of sparse views\n
    sparse_view_depths ```N' x H x W Tensor```: depths of sparse views\n
    sparse_view_positions ```N' x 3 Tensor```: positions of sparse views\n
    """

    def __init__(self, data_desc_path: str):
        """
        Initialize data loader for light field synthesis task

        The data description file is a JSON file with following fields:

        - lf: string, the path of light field image
        - lf_depth: string, the path of light field depth image
        - n_views: { "x",  "y" }, columns and rows of views
        - cam_params: { "f", "c" }, the focal and center of camera (in normalized image space)
        - depth_range: [ min, max ], the range of depth in depth maps
        - depth_layers: int, number of layers in depth maps
        - view_positions: [ [ x, y, z ], ... ], positions of views

        :param data_desc_path: path to the data description file
        """
        self.data_dir = data_desc_path.rsplit('/', 1)[0] + '/'
        with open(data_desc_path, 'r', encoding='utf-8') as file:
            self.data_desc = json.loads(file.read())
        self.n_views = (self.data_desc['n_views']
                        ['y'], self.data_desc['n_views']['x'])
        self.num_views = self.n_views[0] * self.n_views[1]
        self.view_images = ReadLightField(
            self.data_dir + self.data_desc['lf'], self.n_views, True)
        self.view_depths = DecodeDepth(ReadLightField(
            self.data_dir + self.data_desc['lf_depth'], self.n_views, True))
        self.cam_params = self.data_desc['cam_params']
        self.depth_range = self.data_desc['depth_range']
        self.depth_layers = self.data_desc['depth_layers']
        self.view_positions = torch.tensor(self.data_desc['view_positions'])
        _, self.sparse_view_images, self.sparse_view_depths, self.sparse_view_positions \
            = self._GetCornerViews()
        self.diopter_of_layers = self._GetDiopterOfLayers()

    def __len__(self):
        return self.num_views

    def __getitem__(self, idx):
        return idx, self.view_images[idx], self.view_depths[idx], self.view_positions[idx]

    def _GetCornerViews(self):
        corner_selector = torch.zeros(self.num_views, dtype=torch.bool)
        corner_selector[0] = corner_selector[self.n_views[1] - 1] \
            = corner_selector[self.num_views - self.n_views[1]] \
            = corner_selector[self.num_views - 1] = True
        return self.__getitem__(corner_selector)

    def _GetDiopterOfLayers(self) -> List[float]:
        diopter_range = (1 / self.depth_range[1], 1 / self.depth_range[0])
        step = (diopter_range[1] - diopter_range[0]) / (self.depth_layers - 1)
        diopter_of_layers = [diopter_range[0] + step * i for i in range(self.depth_layers)]
        diopter_of_layers.insert(0, 0)
        return diopter_of_layers
