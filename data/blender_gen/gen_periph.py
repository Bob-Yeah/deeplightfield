import bpy
import math
import json
import os
import math
import numpy as np
from itertools import product


scene = bpy.context.scene
cam_obj = scene.camera
cam = cam_obj.data
scene.cycles.device = 'GPU'

dataset_name = 'train'
tbox = [0.7, 0.7, 0.7]
rbox = [300, 120]

dataset_desc = {
    'view_file_pattern': '%s/view_%%04d.png' % dataset_name,
    "gl_coord": True,
    'view_res': {
        'x': 512,
        'y': 512
    },
    'cam_params': {
        'fov': 60.0,
        'cx': 0.5,
        'cy': 0.5,
        'normalized': True
    },
    'range': {
        'min': [-tbox[0] / 2, -tbox[1] / 2, -tbox[2] / 2, -rbox[0] / 2, -rbox[1] / 2],
        'max': [tbox[0] / 2, tbox[1] / 2, tbox[2] / 2, rbox[0] / 2, rbox[1] / 2]
    },
    'samples': [5, 5, 5, 6, 3],
    #'samples': [2000],
    'view_centers': [],
    'view_rots': []
}
data_desc_file = f'output/{dataset_name}.json'

if not os.path.exists('output'):
    os.mkdir('output')

if os.path.exists(data_desc_file):
    with open(data_desc_file, 'r') as fp:
        dataset_desc.update(json.load(fp))
with open(data_desc_file, 'w') as fp:
    json.dump(dataset_desc, fp, indent=4)

# Output resolution
scene.render.resolution_x = dataset_desc['view_res']['x']
scene.render.resolution_y = dataset_desc['view_res']['y']

# Field of view
cam.lens_unit = 'FOV'
cam.angle = math.radians(dataset_desc['cam_params']['fov'])
cam.dof.use_dof = False


def add_sample(i, x, y, z, rx, ry, render_only=False):
    cam_obj.location = [x, y, z]
    cam_obj.rotation_euler = [math.radians(ry), math.radians(rx), 0]
    scene.render.filepath = 'output/' + dataset_desc['view_file_pattern'] % i
    bpy.ops.render.render(write_still=True)
    if not render_only:
        dataset_desc['view_centers'].append(list(cam_obj.location))
        dataset_desc['view_rots'].append([rx, ry])
        with open(data_desc_file, 'w') as fp:
            json.dump(dataset_desc, fp, indent=4)

for i in range(len(dataset_desc['view_centers'])):
    if not os.path.exists('output/' + dataset_desc['view_file_pattern'] % i):
        add_sample(i, *dataset_desc['view_centers'][i], *dataset_desc['view_rots'][i], render_only=True)

start_view = len(dataset_desc['view_centers'])
if len(dataset_desc['samples']) == 1:
    range_min = np.array(dataset_desc['range']['min'])
    range_max = np.array(dataset_desc['range']['max'])
    samples = (range_max - range_min) * np.random.rand(dataset_desc['samples'][0], 5) + range_min
    for i in range(start_view, dataset_desc['samples'][0]):
        add_sample(i, *list(samples[i]))
else:
    ranges = [
        np.linspace(dataset_desc['range']['min'][i],
                    dataset_desc['range']['max'][i],
                    dataset_desc['samples'][i])
        for i in range(0, 3)
    ] + [
        np.linspace(dataset_desc['range']['min'][i],
                    dataset_desc['range']['max'][i],
                    dataset_desc['samples'][i])
        for i in range(3, 5)
    ]

    i = 0
    for x, y, z, rx, ry in product(*ranges):
        if i >= start_view:
            add_sample(i, x, y, z, rx, ry)
        i += 1
