import sys
import os
import torch

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import img
from utils import misc

data_dir = '/home/dengnc/deep_view_syn/data/__7_challenge/classroom_r360x80_t0.3'
in_set = f'{data_dir}/train_depth'
out_set = f'{data_dir}/train_depth_low'

img_names = os.listdir(in_set)

os.chdir(in_set)
depthmaps = img.load(img_names)
depthmaps = torch.floor((depthmaps * 16)) / 16

misc.create_dir(out_set)
os.chdir(out_set)
img.save(depthmaps, img_names)