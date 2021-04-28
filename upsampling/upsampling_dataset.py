import os
import torch
import torchvision.transforms.functional as trans_f
from utils import device
from utils import color
from utils import img


class UpsamplingDataset(torch.utils.data.dataset.Dataset):
    """
    Dataset for upsampling task

    """

    def __init__(self, data_dir: str, input_patt: str, gt_patt: str,
                 c: int, load_once: bool = True):
        """
        Initialize dataset for upsampling task

        :param data_dir: directory of dataset
        :param input_patt: file pattern for input (low resolution) images
        :param gt_patt: file pattern for ground truth (high resolution) images
        :param load_once: load all samples to current device at once to accelerate 
            training, suitable for small dataset
        :param load_gt: whether to load ground truth images
        """
        self.input_patt = os.path.join(data_dir, input_patt)
        self.gt_patt = os.path.join(data_dir, gt_patt) if gt_patt != None else None
        self.n = len(list(filter(
            lambda file_name: os.path.exists(file_name),
            [self.input_patt % i for i in range(
                len(os.listdir(os.path.dirname(self.input_patt))))]
        )))
        self.load_once = load_once
        self.load_gt = self.gt_patt != None
        self.color = c
        self.input = img.load([self.input_patt % i for i in range(self.n)]) \
            .to(device.default()) if self.load_once else None
        self.gt = img.load([self.gt_patt % i for i in range(self.n)]) \
            .to(device.default()) if self.load_once and self.load_gt else None
        if self.color == color.GRAY:
            self.input = trans_f.rgb_to_grayscale(self.input)
            self.gt = trans_f.rgb_to_grayscale(self.gt) \
                if self.gt != None else None
        elif self.color == color.YCbCr:
            self.input = color.rgb2ycbcr(self.input)
            self.gt = color.rgb2ycbcr(self.gt) if self.gt != None else None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.load_once:
            return idx, self.input[idx], self.gt[idx] if self.load_gt else False
        if isinstance(idx, torch.Tensor):
            input = img.load([self.input_patt % i for i in idx])
            gt = img.load([self.gt_patt % i for i in idx]) if self.load_gt else False
        else:
            input = img.load([self.input_patt % idx])
            gt = img.load([self.gt_patt % idx]) if self.load_gt else False
        if self.color == color.GRAY:
            input = trans_f.rgb_to_grayscale(input)
            gt = trans_f.rgb_to_grayscale(gt) if isinstance(gt, torch.Tensor) else False
            return idx, input, gt
        elif self.color == color.YCbCr:
            input = color.rgb2ycbcr(input)
            gt = color.rgb2ycbcr(gt) if isinstance(gt, torch.Tensor) else False
            return idx, input, gt