import json
import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import misc

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rot-range', nargs='+', type=int)
parser.add_argument('-t', '--trans-range', nargs='+', type=float)
parser.add_argument('-k', '--trainset-ratio', type=float, default=0.7)
parser.add_argument('dataset', type=str)
args = parser.parse_args()


data_desc_path = args.dataset
data_desc_name = os.path.splitext(os.path.basename(data_desc_path))[0]
data_dir = os.path.dirname(data_desc_path) + '/'

with open(data_desc_path, 'r') as fp:
    dataset_desc = json.load(fp)

if args.trans_range:
    trans_range = np.array(args.trans_range)
else:
    trans_range = np.array(dataset_desc['range']['max'][0:3]) - \
        np.array(dataset_desc['range']['min'][0:3])
if args.rot_range:
    rot_range = np.array(args.rot_range)
else:
    rot_range = np.array(dataset_desc['range']['max'][3:5]) - \
        np.array(dataset_desc['range']['min'][3:5])
filter_range = np.concatenate([trans_range, rot_range])

out_data_dir = data_dir + 'r%dx%d_t%.1fx%.1fx%.1f/' % (
    int(rot_range[0]), int(rot_range[1]),
    trans_range[0], trans_range[1], trans_range[2]
)

dataset_version = 0
while True:
    out_trainset_name = f'train_{dataset_version}'
    out_testset_name = f'test_{dataset_version}'
    if not os.path.exists(out_data_dir + out_trainset_name):
        break
    dataset_version += 1


def in_range(val, range): return val >= -range / 2 and val <= range / 2


views = []
for i in range(len(dataset_desc['view_centers'])):
    if in_range(dataset_desc['view_rots'][i][0], rot_range[0]) and \
            in_range(dataset_desc['view_rots'][i][1], rot_range[1]) and \
            in_range(dataset_desc['view_centers'][i][0], trans_range[0]) and \
            in_range(dataset_desc['view_centers'][i][1], trans_range[1]) and \
            in_range(dataset_desc['view_centers'][i][2], trans_range[2]):
        views.append(i)

if len(views) < 100:
    print(f'Number of views in range is too small ({len(views)})')
    exit()

views = np.random.permutation(views)
n_train_views = int(len(views) * args.trainset_ratio)
train_views = np.sort(views[:n_train_views])
test_views = np.sort(views[n_train_views:])

print('Train set views: ', len(train_views))
print('Test set views: ', len(test_views))

def create_subset(views, out_desc_name):
    views = views.tolist()
    subset_desc = dataset_desc.copy()
    subset_desc['view_file_pattern'] = \
        f"{out_desc_name}/{dataset_desc['view_file_pattern'].split('/')[-1]}"
    subset_desc['range'] = {
        'min': list(-filter_range / 2),
        'max': list(filter_range / 2)
    }
    subset_desc['samples'] = [int(len(views))]
    subset_desc['views'] = views
    subset_desc['view_centers'] = np.array(dataset_desc['view_centers'])[views].tolist()
    subset_desc['view_rots'] = np.array(dataset_desc['view_rots'])[views].tolist()

    with open(os.path.join(out_data_dir, f'{out_desc_name}.json'), 'w') as fp:
        json.dump(subset_desc, fp, indent=4)
    misc.create_dir(os.path.join(out_data_dir, out_desc_name))
    for i in range(len(views)):
        os.symlink(os.path.join('../../', dataset_desc['view_file_pattern'] % views[i]),
                   os.path.join(out_data_dir, subset_desc['view_file_pattern'] % views[i]))


misc.create_dir(out_data_dir)
create_subset(train_views, out_trainset_name)
create_subset(train_views, out_testset_name)
