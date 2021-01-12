import os
from numpy.core.fromnumeric import trace
from numpy.lib.arraysetops import isin
import torch
import torchvision.transforms.functional as trans_f
from ..my import util
from ..my import device
from ..my import color_mode


class UpsamplingDataset(torch.utils.data.dataset.Dataset):
    """
    Dataset for upsampling task

    """

    def __init__(self, data_dir: str, input_patt: str, gt_patt: str,
                 color: int, load_once: bool = True):
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
        self.color = color
        self.input = util.ReadImageTensor([self.input_patt % i for i in range(self.n)]) \
            .to(device.GetDevice()) if self.load_once else None
        self.gt = util.ReadImageTensor([self.gt_patt % i for i in range(self.n)]) \
            .to(device.GetDevice()) if self.load_once and self.load_gt else None
        if self.color == color_mode.GRAY:
            self.input = trans_f.rgb_to_grayscale(self.input)
            self.gt = trans_f.rgb_to_grayscale(self.gt) \
                if self.gt != None else None
        elif self.color == color_mode.YCbCr:
            self.input = util.rgb2ycbcr(self.input)
            self.gt = util.rgb2ycbcr(self.gt) if self.gt != None else None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.load_once:
            return idx, self.input[idx], self.gt[idx] if self.load_gt else False
        if isinstance(idx, torch.Tensor):
            input = util.ReadImageTensor([self.input_patt % i for i in idx])
            gt = util.ReadImageTensor([self.gt_patt % i for i in idx]) if self.load_gt else False
        else:
            input = util.ReadImageTensor([self.input_patt % idx])
            gt = util.ReadImageTensor([self.gt_patt % idx]) if self.load_gt else False
        if self.color == color_mode.GRAY:
            input = trans_f.rgb_to_grayscale(input)
            gt = trans_f.rgb_to_grayscale(gt) if isinstance(gt, torch.Tensor) else False
            return idx, input, gt
        elif self.color == color_mode.YCbCr:
            input = util.rgb2ycbcr(input)
            gt = util.rgb2ycbcr(gt) if isinstance(gt, torch.Tensor) else False
            return idx, input, gt