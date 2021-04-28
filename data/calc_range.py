import json
import sys
import os
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import misc

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
args = parser.parse_args()


data_desc_path = args.dataset
data_desc_name = os.path.splitext(os.path.basename(data_desc_path))[0]
data_dir = os.path.dirname(data_desc_path) + '/'

with open(data_desc_path, 'r') as fp:
    dataset_desc = json.load(fp)

centers = np.array(dataset_desc['view_centers'])
t_max = np.max(centers, axis=0)
t_min = np.min(centers, axis=0)
dataset_desc['range'] = {
    'min': [t_min[0], t_min[1], t_min[2], 0, 0],
    'max': [t_max[0], t_max[1], t_max[2], 0, 0]
}
with open(data_desc_path, 'w') as fp:
    json.dump(dataset_desc, fp, indent=4)