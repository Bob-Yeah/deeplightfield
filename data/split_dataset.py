import json
import sys
import os
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import misc

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, default='train1')
parser.add_argument('dataset', type=str)
args = parser.parse_args()


data_desc_path = args.dataset
data_desc_name = os.path.splitext(os.path.basename(data_desc_path))[0]
data_dir = os.path.dirname(data_desc_path) + '/'

with open(data_desc_path, 'r') as fp:
    dataset_desc = json.load(fp)

indices = torch.arange(len(dataset_desc['view_centers'])).view(dataset_desc['samples'])

idx = 0
'''
for i in range(3):
    for j in range(2):
        out_desc_name = f'part{idx:d}'
        out_desc = dataset_desc.copy()
        out_desc['view_file_pattern'] = f'{out_desc_name}/view_%04d.png'
        n_x = out_desc['samples'][3] // 3
        n_y = out_desc['samples'][4] // 2
        views = indices[..., i * n_x:(i + 1) * n_x, j * n_y:(j + 1) * n_y].flatten().tolist()
        out_desc['samples'] = [len(views)]
        out_desc['views'] = views
        out_desc['view_centers'] = np.array(dataset_desc['view_centers'])[views].tolist()
        out_desc['view_rots'] = np.array(dataset_desc['view_rots'])[views].tolist()
        with open(os.path.join(data_dir, f'{out_desc_name}.json'), 'w') as fp:
            json.dump(out_desc, fp, indent=4)
        misc.create_dir(os.path.join(data_dir, out_desc_name))
        for k in range(len(views)):
            os.symlink(os.path.join('..', dataset_desc['view_file_pattern'] % views[k]),
                    os.path.join(data_dir, out_desc['view_file_pattern'] % views[k]))
        idx += 1
'''

'''
for xi in range(0, 4, 2):
    for yi in range(0, 4, 2):
        for zi in range(0, 4, 2):
            out_desc_name = f'part{idx:d}'
            out_desc = dataset_desc.copy()
            out_desc['view_file_pattern'] = f'{out_desc_name}/view_%04d.png'
            views = indices[xi:xi + 2, yi:yi + 2, zi:zi + 2].flatten().tolist()
            out_desc['samples'] = [len(views)]
            out_desc['views'] = views
            out_desc['view_centers'] = np.array(dataset_desc['view_centers'])[views].tolist()
            out_desc['view_rots'] = np.array(dataset_desc['view_rots'])[views].tolist()
            with open(os.path.join(data_dir, f'{out_desc_name}.json'), 'w') as fp:
                json.dump(out_desc, fp, indent=4)
            misc.create_dir(os.path.join(data_dir, out_desc_name))
            for k in range(len(views)):
                os.symlink(os.path.join('..', dataset_desc['view_file_pattern'] % views[k]),
                           os.path.join(data_dir, out_desc['view_file_pattern'] % views[k]))
            idx += 1
'''
from itertools import product
out_desc_name = args.output
out_desc = dataset_desc.copy()
out_desc['view_file_pattern'] = f"{out_desc_name}/{dataset_desc['view_file_pattern'].split('/')[-1]}"
views = []
for idx in product([0, 2, 4], [0, 2, 4], [0, 2, 4], list(range(9)), [1]):#, [0, 2, 3, 5], [1, 2, 3, 4]):
    views += indices[idx].flatten().tolist()
out_desc['samples'] = [len(views)]
out_desc['views'] = views
out_desc['view_centers'] = np.array(dataset_desc['view_centers'])[views].tolist()
out_desc['view_rots'] = np.array(dataset_desc['view_rots'])[views].tolist()
with open(os.path.join(data_dir, f'{out_desc_name}.json'), 'w') as fp:
    json.dump(out_desc, fp, indent=4)
misc.create_dir(os.path.join(data_dir, out_desc_name))
for k in range(len(views)):
    os.symlink(os.path.join('..', dataset_desc['view_file_pattern'] % views[k]),
               os.path.join(data_dir, out_desc['view_file_pattern'] % views[k]))
