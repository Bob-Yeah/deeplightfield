import json
import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

from utils import seqs
from utils import misc
from utils.constants import *

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rot-range', nargs='+', type=int)
parser.add_argument('-t', '--trans-range', nargs='+', type=float)
parser.add_argument('--fov', type=float)
parser.add_argument('--res', type=str)
parser.add_argument('--gl', action='store_true')
parser.add_argument('-s', '--seq', type=str, required=True)
parser.add_argument('-n', '--views', type=int, required=True)
parser.add_argument('-o', '--out-desc', type=str)
parser.add_argument('--ref', type=str)
parser.add_argument('dataset', type=str)
args = parser.parse_args()


data_dir = args.dataset
misc.create_dir(data_dir)
out_desc_path = os.path.join(data_dir, (args.out_desc if args.out_desc else f"{args.seq}.json"))

if args.ref:
    with open(os.path.join(data_dir, args.ref), 'r') as fp:
        ref_desc = json.load(fp)
else:
    if not args.trans_range or not args.rot_range or not args.fov or not args.res:
        print('-r, -t, --fov, --res options are required if --ref is not specified')
        exit(-1)
    ref_desc = None

if args.trans_range:
    trans_range = np.array(list(args.trans_range) * 3 if len(args.trans_range) == 1
                           else args.trans_range)
else:
    trans_range = np.array(ref_desc['range']['max'][0:3]) - \
        np.array(ref_desc['range']['min'][0:3])
if args.rot_range:
    rot_range = np.array(list(args.rot_range) * 2 if len(args.rot_range) == 1
                         else args.rot_range)
else:
    rot_range = np.array(ref_desc['range']['max'][3:5]) - \
        np.array(ref_desc['range']['min'][3:5])
filter_range = np.concatenate([trans_range, rot_range])

if args.fov:
    cam_params = {
        'fov': args.fov,
        'cx': 0.5,
        'cy': 0.5,
        'normalized': True
    }
else:
    cam_params = ref_desc['cam_params']

if args.res:
    res = tuple(int(s) for s in args.res.split('x'))
    res = {'x': res[0], 'y': res[1]}
else:
    res = ref_desc['view_res']

if args.seq == 'helix':
    centers, rots = seqs.helix(trans_range, 4, args.views)
elif args.seq == 'scan_around':
    centers, rots = seqs.scan_around(trans_range, 1, args.views)
elif args.seq == 'look_around':
    centers, rots = seqs.look_around(trans_range, args.views)

rots *= 180 / PI
gl = args.gl or ref_desc.get('gl_coord')
if gl:
    centers[:, 2] *= -1
    rots[:, 0] *= -1

dataset_desc = {
    'gl_coord': gl,
    'view_res': res,
    'cam_params': cam_params,
    'range': {
        'min': (-0.5 * filter_range).tolist(),
        'max': (0.5 * filter_range).tolist()
    },
    'samples': [args.views],
    'view_centers': centers.tolist(),
    'view_rots': rots.tolist()
}

with open(out_desc_path, 'w') as fp:
    json.dump(dataset_desc, fp, indent=4)
