import sys
import os
sys.path.append(os.path.abspath(sys.path[0] + '/../'))
__package__ = "deep_view_syn"

import argparse
from PIL import Image
from .my import util


def batch_scale(src, target, size):
    util.CreateDirIfNeed(target)
    for file_name in os.listdir(src):
        postfix = os.path.splitext(file_name)[1]
        if postfix == '.jpg' or postfix == '.png':
            im = Image.open(os.path.join(src, file_name))
            im = im.resize(size)
            im.save(os.path.join(target, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str,
                        help='Source directory.')
    parser.add_argument('target', type=str,
                        help='Target directory.')
    parser.add_argument('--width', type=int,
                        help='Width of output images (pixel)')
    parser.add_argument('--height', type=int,
                        help='Height of output images (pixel)')
    opt = parser.parse_args()
    batch_scale(opt.src, opt.target, (opt.width, opt.height))
