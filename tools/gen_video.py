import sys
import os
import argparse
import torch
import cv2
from torchvision.io.video import write_video
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(sys.path[0] + '/../'))

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0,
                    help='Which CUDA device to use.')
parser.add_argument('--view-file', type=str, default='hmd.csv')
parser.add_argument('--fps', type=int, default=50)
parser.add_argument('--add-hint', action='store_true')
parser.add_argument('--output-video', action='store_true')
parser.add_argument('scene', type=str)
opt = parser.parse_args()

# Select device
torch.cuda.set_device(opt.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())
torch.autograd.set_grad_enabled(False)

from data.spherical_view_syn import *
from configs.spherical_view_syn import SphericalViewSynConfig
from utils import netio
from utils import misc
from utils import img
from utils import device
from utils import view
from utils import sphere
from components.gen_final import GenFinal
from utils.progress_bar import progress_bar


def load_net(path):
    config = SphericalViewSynConfig()
    config.from_id(path[:-4])
    config.SAMPLE_PARAMS['perturb_sample'] = False
    # config.print()
    net = config.create_net().to(device.default())
    netio.load(path, net)
    return net


def find_file(prefix):
    for path in os.listdir():
        if path.startswith(prefix):
            return path
    return None

rot_range = {
    'gas': [-15, 15, -15, 15],
    'mc': [-20, 20, -20, 20],
    'bedroom': [-40, 40, -40, 40],
    'lobby': [-20, 20, -20, 20],
    'gallery': [-50, 50, -10, 10]
}
trans_range = [-0.15, 0.15, -0.15, 0.15, -0.15, 0.15]
def clamp_gaze(gaze):
    return gaze
    scoord = sphere.cartesian2spherical(gaze)


def load_views(data_desc_file) -> Tuple[view.Trans, torch.Tensor]:
    with open(data_desc_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        n = len(lines) // 7
        gazes = torch.empty(n * 2, 3)
        views = torch.empty(n * 2, 4, 4)
        view_idx = 0
        for i in range(0, len(lines), 7):
            gazes[view_idx * 2] = clamp_gaze(torch.tensor([
                float(str) for str in lines[i + 1].split(',')
            ]))
            gazes[view_idx * 2 + 1] = clamp_gaze(torch.tensor([
                float(str) for str in lines[i + 2].split(',')
            ]))
            views[view_idx * 2] = torch.tensor([
                float(str) for str in lines[i + 3].split(',')
            ]).view(4, 4)
            views[view_idx * 2 + 1] = torch.tensor([
                float(str) for str in lines[i + 4].split(',')
            ]).view(4, 4)
            view_idx += 1
        gazes = gazes.to(device.default())
        views = views.to(device.default())
    return view.Trans(views[:, :3, 3], views[:, :3, :3]), gazes


fov_list = [20, 45, 110]
res_list = [(128, 128), (256, 256), (256, 230)]  # (192,256)]
res_full = (1600, 1440)


scenes = {
    'gas': '__0_user_study/us_gas_all_in_one',
    'mc': '__0_user_study/us_mc_all_in_one',
    'bedroom': 'bedroom_all_in_one',
    'gallery': 'gallery_all_in_one',
    'lobby': 'lobby_all_in_one'
}
os.chdir(sys.path[0] + '/../data/' + scenes[opt.scene])
print('Change working directory to ', os.getcwd())

fovea_net = load_net(find_file('fovea'))
periph_net = load_net(find_file('periph'))

# Load Dataset
views, gazes = load_views(opt.view_file)
n_views = views.size()[0] // 2
print('Dataset loaded.')
print('views:', n_views)

gen = GenFinal(fov_list, res_list, res_full, fovea_net, periph_net,
               device=device.default())
gaze_centers = gen.full_cam.proj(gazes, center_as_origin=True)

videodir = sys.path[0] + '/../data/__3_video/'
inferoutdir = videodir + \
    '%s_%s/' % (opt.scene, os.path.splitext(opt.view_file)[0])
hintoutdir = videodir + \
    '%s_%s_with_hint/' % (opt.scene, os.path.splitext(opt.view_file)[0])
hint = img.load(sys.path[0] + '/fovea_hint.png', rgb_only=False)


def add_hint(img, center):
    fovea_origin = (
        int(center[0]) + res_full[1] // 2 - hint.size(-1) // 2,
        int(center[1]) + res_full[0] // 2 - hint.size(-2) // 2
    )
    fovea_region = (
        ...,
        slice(fovea_origin[1], fovea_origin[1] + hint.size(-2)),
        slice(fovea_origin[0], fovea_origin[0] + hint.size(-1)),
    )
    img[fovea_region] = img[fovea_region] * (1 - hint[:, 3:]) + \
        hint[:, :3] * hint[:, 3:]


imgs = torch.empty(n_views, 3, res_full[0], res_full[1] * 2)

if opt.add_hint and os.path.exists(inferoutdir + '/view0000.png'):
    for view_idx in range(n_views):
        img = img.load(inferoutdir + '/view%04d.png' % view_idx)
        left_center = (gaze_centers[view_idx * 2][0].item(),
                       gaze_centers[view_idx * 2][1].item())
        right_center = (gaze_centers[view_idx * 2 + 1][0].item(),
                        gaze_centers[view_idx * 2 + 1][1].item())
        add_hint(img, left_center)
        add_hint(img[..., res_full[1]:], right_center)
        imgs[view_idx:view_idx + 1] = img
        progress_bar(view_idx, n_views, 'Frame %4d processed' % view_idx)
else:
    for view_idx in range(n_views):
        left_center = (gaze_centers[view_idx * 2][0].item(),
                       gaze_centers[view_idx * 2][1].item())
        right_center = (gaze_centers[view_idx * 2 + 1][0].item(),
                        gaze_centers[view_idx * 2 + 1][1].item())
        left_view = views.get(view_idx * 2)
        right_view = views.get(view_idx * 2 + 1)
        mono_trans = view.Trans((left_view.t + right_view.t) / 2, left_view.r)
        left_image = gen.gen(left_center, left_view,
                             mono_trans=mono_trans)['blended'].cpu()
        right_image = gen.gen(right_center, right_view,
                              mono_trans=mono_trans)['blended'].cpu()
        if opt.add_hint:
            add_hint(left_image, left_center)
            add_hint(right_image, right_center)
        imgs[view_idx:view_idx + 1] = torch.cat([left_image, right_image], -1)
        progress_bar(view_idx, n_views, 'Frame %4d inferred' % view_idx)


if opt.output_video:
    video_file = videodir + '%s_%s_%s.mp4' % (
        opt.scene, os.path.splitext(opt.view_file)[0],
        'with_hint' if opt.add_hint else '')
    print('Write video file ' + os.path.abspath(video_file))
    imgs = img.torch2np(imgs)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(video_file,fourcc, 50.0, (1440*2, 1600))
    for view_idx in range(n_views):
        out.write(imgs[view_idx])
    out.release()
else:
    outdir = hintoutdir if opt.add_hint else inferoutdir
    misc.create_dir(outdir)
    for view_idx in range(n_views):
        img.save(imgs[view_idx], outdir + 'view%04d.png' % view_idx)
        progress_bar(view_idx, n_views, 'Frame %4d saved' % view_idx)
