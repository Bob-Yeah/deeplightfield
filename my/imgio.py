from typing import List, NoReturn
import torch
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io.image import read_image
from torchvision.utils import save_image


def ReadImages(*args, paths: List[str] = None, dtype=torch.float) -> torch.Tensor:
    raise NotImplementedError('The method has bug. Use util.ReadImageTensor instead')
    if not paths:
        paths = args
    images = torch.stack([read_image(path) for path in paths], dim=0)
    return convert_image_dtype(images, dtype)


def SaveImages(input, *args, paths: List[str] = None) -> NoReturn:
    raise NotImplementedError('The method has bug. Use util.WriteImageTensor instead')
    if not paths:
        paths = args
    if input.size(0) != len(paths):
        raise ValueError('batch size of input images is not same as length of paths')
    for i, path in enumerate(range(paths)):
        save_image(input[i], path)