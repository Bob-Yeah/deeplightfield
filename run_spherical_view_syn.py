import os
import sys
import argparse
import shutil
from typing import Mapping
from utils.constants import TINY_FLOAT
import torch
import torch.optim
import math
import time
from tensorboardX import SummaryWriter
from torch import nn
from numpy.core.numeric import NaN

parser = argparse.ArgumentParser()
# Arguments for train >>>
parser.add_argument('-c', '--config', type=str,
                    help='Net config files')
parser.add_argument('-i', '--config-id', type=str,
                    help='Net config id')
parser.add_argument('-e', '--epochs', type=int, default=200,
                    help='Max epochs for train')
parser.add_argument('-n', '--prev-net', type=str)
# Arguments for test >>>
parser.add_argument('-r', '--output-res', type=str,
                    help='Output resolution')
parser.add_argument('-o', '--output', nargs='+', type=str, default=['perf', 'color'],
                    help='Specify what to output (perf, color, depth, all)')
parser.add_argument('--output-type', type=str, default='image',
                    help='Specify the output type (image, video, debug)')
# Other arguments >>>
parser.add_argument('-t', '--test', action='store_true',
                    help='Start in test mode')
parser.add_argument('-m', '--model', type=str,
                    help='The model file to load for continue train or test')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='Which CUDA device to use.')
parser.add_argument('-l', '--log-redirect', action='store_true',
                    help='Is log redirected to file?')
parser.add_argument('-p', '--prompt', action='store_true',
                    help='Interactive prompt mode')
parser.add_argument('dataset', type=str,
                    help='Dataset description file')
args = parser.parse_args()


torch.cuda.set_device(args.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())


from utils import netio
from utils import misc
from utils import device
from utils import img
from utils import interact
from utils.progress_bar import progress_bar
from utils.perf import Perf
from data.spherical_view_syn import *
from data.loader import FastDataLoader
from configs.spherical_view_syn import SphericalViewSynConfig
from loss.ssim import ssim


data_desc_path = args.dataset if args.dataset.endswith('.json') \
    else os.path.join(args.dataset, 'train.json')
data_desc_name = os.path.splitext(os.path.basename(data_desc_path))[0]
data_dir = os.path.dirname(data_desc_path) + '/'
config = SphericalViewSynConfig()
BATCH_SIZE = 4096
SAVE_INTERVAL = 10
TEST_BATCH_SIZE = 1
TEST_MAX_RAYS = 32768 // 2

# Toggles
ROT_ONLY = False
EVAL_TIME_PERFORMANCE = False
# ========
#ROT_ONLY = True
#EVAL_TIME_PERFORMANCE = True


def get_model_files(datadir):
    model_files = []
    for root, _, files in os.walk(datadir):
        model_files += [
            os.path.join(root, file).replace(datadir, '')
            for file in files if file.endswith('.pth')
        ]
    return model_files


def set_outputs(args, outputs_str: str):
    args.output = [s.strip() for s in outputs_str.split(',')]


if not args.test:
    print('Start in train mode.')
    if args.prompt:  # 2.1 Prompt max epochs
        args.epochs = interact.input_ex('Max epochs:', interact.input_to_int(min=1),
                                        default=200)
    epochRange = range(1, args.epochs + 1)
    if args.prompt:  # 2.2 Prompt continue train
        model_files = get_model_files(data_dir)
        args.model = interact.input_enum('Continue train on model:', model_files,
                                         err_msg='No such model file', default='')
    if args.model:
        cont_model = os.path.join(data_dir, args.model)
        model_name = os.path.splitext(os.path.basename(cont_model))[0]
        epochRange = range(int(model_name[12:]) + 1, epochRange.stop)
        run_dir = os.path.dirname(cont_model) + '/'
        run_id = os.path.basename(run_dir[:-1])
        config.from_id(run_id)
    else:
        if args.prompt:  # 2.3 Prompt config file and additional config items
            config_files = [
                f[:-3] for f in os.listdir('configs')
                if f.endswith('.py') and f != 'spherical_view_syn.py'
            ]
            args.config = interact.input_enum('Specify config file:', config_files,
                                              err_msg='No such config file', default='')
            args.config_id = interact.input_ex('Specify custom config items:',
                                               default='')
        if args.config:
            config.load(os.path.join('configs', args.config + '.py'))
        if args.config_id:
            config.from_id(args.config_id)
        run_id = config.to_id()
        run_dir = data_dir + run_id + '/'
    log_dir = run_dir + 'log/'
else:  # Test mode
    print('Start in test mode.')
    if args.prompt:  # 3. Prompt test model, output resolution, output mode
        model_files = get_model_files(data_dir)
        args.model = interact.input_enum('Specify test model:', model_files,
                                         err_msg='No such model file')
        args.output_res = interact.input_ex('Specify output resolution:',
                                            default='')
        set_outputs(args, interact.input_ex('Specify the outputs | [perf,color,depth,layers]/all:',
                                            default='perf,color'))
        args.output_type = interact.input_enum('Specify the output type | image/video:',
                                               ['image', 'video'],
                                               err_msg='Wrong output type',
                                               default='image')
    test_model_path = os.path.join(data_dir, args.model)
    test_model_name = os.path.splitext(os.path.basename(test_model_path))[0]
    run_dir = os.path.dirname(test_model_path) + '/'
    run_id = os.path.basename(run_dir[:-1])
    config.from_id(run_id)
    config.SAMPLE_PARAMS['perturb_sample'] = False
    args.output_res = tuple(int(s) for s in args.output_res.split('x')) \
        if args.output_res else None
    output_dir = f"{run_dir}output_{int(test_model_name.split('_')[-1])}"
    output_dataset_id = '%s%s' % (
        data_desc_name,
        '_%dx%d' % (args.output_res[0], args.output_res[1]) if args.output_res else '')
    args.output_flags = {
        item: item in args.output or 'all' in args.output
        for item in ['perf', 'color', 'depth', 'layers']
    }


config.print()
print("run dir: ", run_dir)

# Initialize model
model = config.create_net().to(device.default())
loss_mse = nn.MSELoss().to(device.default())


if args.prev_net:
    prev_net_config_id = os.path.split(args.prev_net)[-2]
    prev_net_config = SphericalViewSynConfig()
    prev_net_config.from_id(prev_net_config_id)
    prev_net = prev_net_config.create_net().to(device.default())
    netio.load(args.prev_net, prev_net)
    model.prev_net = prev_net


toggle_show_dir = False
last_toggle_time = 0

from nets.cnerf_v3 import CNerf
from nets.nerf_depth import NerfDepth
is_dnerf = isinstance(model, NerfDepth)
is_cnerf = isinstance(model, CNerf)


def train_loop(data_loader, optimizer, perf, writer, epoch, iters):
    global toggle_show_dir
    global last_toggle_time
    dataset: SphericalViewSynDataset = data_loader.dataset
    sub_iters = 0
    iters_in_epoch = len(data_loader)
    loss_min = 1e5
    loss_max = 0
    loss_avg = 0
    perf1 = Perf(args.log_redirect, True)
    for idx, gt, rays_o, rays_d in data_loader:
        if is_dnerf:
            rays_depth = dataset.patched_depths[idx] if dataset.load_depths else None
            rays_bins = dataset.patched_bins[idx] if dataset.load_bins else None
            perf.checkpoint("Load")

            out = model(rays_o, rays_d, rays_depth, rays_bins)
            if isinstance(out, torch.Tensor):
                out = {'color': out}
            if isinstance(out, Mapping):
                out = [out]
            perf.checkpoint("Forward")

            optimizer.zero_grad()
            loss_value = loss_mse(out[0]['color'], gt)
            for i in range(1, len(out)):
                loss_value += loss_mse(out[i]['color'], gt)
        elif is_cnerf:
            rays_weights = model.bin_weights.flatten(0, 2)[idx]
            perf.checkpoint("Load")

            out = model(rays_o, rays_d, rays_weights)
            if isinstance(out, torch.Tensor):
                out = {'color': out}
            if isinstance(out, Mapping):
                out = [out]
            perf.checkpoint("Forward")

            optimizer.zero_grad()
            loss_value = loss_mse(out[0]['color'], gt)
            for i in range(1, len(out)):
                loss_value += loss_mse(out[i]['color'], gt)
        else:
            gt_disp = torch.reciprocal(dataset.patched_depths[idx]) if config.DEPTH_REF else None
            perf.checkpoint("Load")

            out = model(rays_o, rays_d, ret_depth=config.DEPTH_REF)
            if isinstance(out, torch.Tensor):
                out = {'color': out}
            if isinstance(out, Mapping):
                out = [out]
            perf.checkpoint("Forward")

            optimizer.zero_grad()
            loss_value = loss_mse(out[0]['color'], gt)
            for i in range(1, len(out)):
                loss_value += loss_mse(out[i]['color'], gt)
            if config.DEPTH_REF:
                disp_loss_value = loss_mse(torch.reciprocal(out[0]['depth'] + TINY_FLOAT), gt_disp)
                for i in range(1, len(out)):
                    disp_loss_value += loss_mse(torch.reciprocal(
                        out[i]['depth'] + TINY_FLOAT), gt_disp)
                disp_loss_value = disp_loss_value / math.pow(
                    1 / dataset.depth_range[0] - 1 / dataset.depth_range[1], 2)
            else:
                disp_loss_value = 0
            loss_value += disp_loss_value
        perf.checkpoint("Compute loss")

        loss_value.backward()
        perf.checkpoint("Backward")

        optimizer.step()
        perf.checkpoint("Update")

        loss_value = loss_value.item()
        loss_min = min(loss_min, loss_value)
        loss_max = max(loss_max, loss_value)
        loss_avg = (loss_avg * sub_iters + loss_value) / (sub_iters + 1)
        if not args.log_redirect:
            progress_bar(sub_iters, iters_in_epoch,
                         f"Loss: {loss_value:.2e} ({loss_min:.2e}/{loss_avg:.2e}/{loss_max:.2e})",
                         f"Epoch {epoch:<3d}")
            current_time = time.time()
            if last_toggle_time == 0:
                last_toggle_time = current_time
            if current_time - last_toggle_time > 3:
                toggle_show_dir = not toggle_show_dir
                last_toggle_time = current_time
            if toggle_show_dir:
                sys.stdout.write(f'Epoch {epoch:<3d} [ {run_dir}    ]\r')

        # Write tensorboard logs.
        writer.add_scalar("loss mse", loss_value, iters)
        # if patch and iters % 100 == 0:
        #    output_vs_gt = torch.cat([out[0:4], gt[0:4]], 0).detach()
        #    writer.add_image("Output_vs_gt", torchvision.utils.make_grid(
        #        output_vs_gt, nrow=4).cpu().numpy(), iters)

        iters += 1
        sub_iters += 1
    if args.log_redirect:
        perf1.checkpoint('Epoch %d (%.2e/%.2e/%.2e)' %
                         (epoch, loss_min, loss_avg, loss_max), True)
    return iters


def save_checkpoint(epoch, iters):
    for i in range(1, epoch):
        if (i < epoch // 50 * 50 and i % 50 != 0 or i % 10 != 0) and \
                os.path.exists(f'{run_dir}model-epoch_{i}.pth'):
            os.remove(f'{run_dir}model-epoch_{i}.pth')
    netio.save(f'{run_dir}model-epoch_{epoch}.pth', model, iters, print_log=False)


def train():
    # 1. Initialize data loader
    print("Load dataset: " + data_desc_path)
    dataset = SphericalViewSynDataset(data_desc_path, c=config.COLOR, load_depths=config.DEPTH_REF,
                                      load_bins=config.DEPTH_REF)
    dataset.set_patch_size(1)
    data_loader = FastDataLoader(dataset, BATCH_SIZE, shuffle=True, pin_memory=True)

    if is_cnerf:
        model.set_depth_maps(dataset.rays_o, dataset.rays_d, dataset.view_depths)

    # 2. Initialize components
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if epochRange.start > 1:
        iters = netio.load(f'{run_dir}model-epoch_{epochRange.start - 1}.pth', model)
    else:
        misc.create_dir(run_dir)
        misc.create_dir(log_dir)
        if config.NORMALIZE:
            for _, _, rays_o, rays_d in data_loader:
                model.update_normalize_range(rays_o, rays_d)
            print('Depth/diopter range: ', model.depth_range)
            print('Angle range: ', model.angle_range.rad2deg())
        iters = 0

    # 3. Train
    model.train()

    perf = Perf(EVAL_TIME_PERFORMANCE, start=True)
    writer = SummaryWriter(log_dir)

    print("Begin training...")
    for epoch in epochRange:
        iters = train_loop(data_loader, optimizer, perf, writer, epoch, iters)
        save_checkpoint(epoch, iters)
    print("Train finished")


def test():
    with torch.no_grad():
        # 1. Load dataset
        print("Load dataset: " + data_desc_path)
        dataset = SphericalViewSynDataset(data_desc_path, res=args.output_res,
                                          load_images=args.output_flags['perf'])
        data_loader = FastDataLoader(dataset, 1, shuffle=False, pin_memory=True)

        # 2. Load trained model
        netio.load(test_model_path, model)
        model.eval()

        # 3. Test on dataset
        print("Begin test, batch size is %d" % TEST_BATCH_SIZE)

        i = 0
        global_offset = 0
        chns = color.chns(config.COLOR)
        n = dataset.n_views
        total_pixels = n * dataset.view_res[0] * dataset.view_res[1]

        out = {}
        if args.output_flags['perf']:
            perf_times = torch.empty(n)
            perf = Perf(True, start=True)
        if args.output_flags['layers']:
            out['layers'] = torch.empty(total_pixels, config.SAMPLE_PARAMS['n_samples'], chns + 1,
                                        device=device.default())
        if args.output_flags['perf'] or args.output_flags['color']:
            out['color'] = torch.empty(total_pixels, chns, device=device.default())
        if args.output_flags['depth']:
            out['depth'] = torch.empty(total_pixels, device=device.default())
            out['bins'] = torch.zeros(total_pixels, 3, device=device.default())

        for _, _, rays_o, rays_d in data_loader:
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)
            n_rays = rays_o.size(0)
            for offset in range(0, n_rays, TEST_MAX_RAYS):
                idx = slice(offset, min(offset + TEST_MAX_RAYS, n_rays))
                global_idx = slice(idx.start + global_offset, idx.stop + global_offset)
                ret = model(rays_o[idx], rays_d[idx],
                            ret_depth=args.output_flags['depth'],
                            debug=args.output_flags['layers'])
                if isinstance(ret, torch.Tensor):
                    ret = {'color': ret}
                if isinstance(ret, list):
                    ret = ret[-1]
                if 'bins' in out:
                    ret['weight'] = ret['weight'].view(-1, ret['weight'].size(-1) // 2, 2).sum(-1)
                    is_local_max = torch.ones_like(ret['weight'], dtype=torch.bool)
                    for delta in range(-3, 0):
                        is_local_max[..., -delta:].logical_and_(
                            ret['weight'][..., -delta:] > ret['weight'][..., :delta])
                    for delta in range(1, 4):
                        is_local_max[..., :-delta].logical_and_(
                            ret['weight'][..., :-delta] > ret['weight'][..., delta:])
                    ret['weight'][is_local_max.logical_not()] = 0
                    vals, idxs = torch.topk(ret['weight'], 3)  # (B, 3)
                    vals = vals / vals.sum(-1, keepdim=True)
                    ret['bins'] = (idxs.to(torch.float) / (ret['weight'].size(-1) - 1)
                                   * 0.5 + 0.5) * (vals > 0.1)

                for key in out:
                    out[key][global_idx] = ret[key]
            if args.output_flags['perf']:
                perf_times[i] = perf.checkpoint()
            progress_bar(i, n, 'Inferring...')
            i += 1
            global_offset += n_rays

        # 4. Save results
        print('Saving results...')
        misc.create_dir(output_dir)

        for key in out:
            shape = [n] + list(dataset.view_res) + list(out[key].size()[1:])
            out[key] = out[key].view(shape)
        if 'color' in out:
            out['color'] = out['color'].permute(0, 3, 1, 2)
        if 'layers' in out:
            # n, y, x, samples, chns -> samples, n, chns, y, x
            out['layers'] = out['layers'].permute(3, 0, 4, 1, 2)
        if 'bins' in out:
            out['bins'] = out['bins'].permute(0, 3, 1, 2)

        if args.output_flags['perf']:
            perf_errors = torch.ones(n) * NaN
            perf_ssims = torch.ones(n) * NaN
            if dataset.view_images != None:
                for i in range(n):
                    perf_errors[i] = loss_mse(dataset.view_images[i], out['color'][i]).item()
                    perf_ssims[i] = ssim(dataset.view_images[i:i + 1],
                                         out['color'][i:i + 1]).item() * 100
            perf_mean_time = torch.mean(perf_times).item()
            perf_mean_error = torch.mean(perf_errors).item()
            perf_name = 'perf_%s_%.1fms_%.2e.csv' % (
                output_dataset_id, perf_mean_time, perf_mean_error)

            # Remove old performance reports
            for file in os.listdir(output_dir):
                if file.startswith(f'perf_{output_dataset_id}'):
                    os.remove(f"{output_dir}/{file}")

            # Save new performance reports
            with open(f"{output_dir}/{perf_name}", 'w') as fp:
                fp.write('View, Time, PSNR, SSIM\n')
                fp.writelines([
                    f'{dataset.view_idxs[i]}, {perf_times[i].item():.2f}, '
                    f'{img.mse2psnr(perf_errors[i].item()):.2f}, {perf_ssims[i].item():.2f}\n'
                    for i in range(n)
                ])

        if args.output_flags['color']:
            if args.output_type == 'video':
                output_file = f"{output_dir}/{output_dataset_id}_color.mp4"
                img.save_video(out['color'], output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_color"
                misc.create_dir(output_subdir)
                img.save(out['color'], [f'{output_subdir}/{i:0>4d}.png' for i in dataset.view_idxs])

        if args.output_flags['depth']:
            colorized_depths = img.colorize_depthmap(
                out['depth'], config.SAMPLE_PARAMS['depth_range'])
            if args.output_type == 'video':
                output_file = f"{output_dir}/{output_dataset_id}_depth.mp4"
                img.save_video(colorized_depths, output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_depth"
                misc.create_dir(output_subdir)
                img.save(colorized_depths, [
                    f'{output_subdir}/{i:0>4d}.png'
                    for i in dataset.view_idxs
                ])
                output_subdir = f"{output_dir}/{output_dataset_id}_bins"
                misc.create_dir(output_subdir)
                img.save(out['bins'], [f'{output_subdir}/{i:0>4d}.png' for i in dataset.view_idxs])

        if args.output_flags['layers']:
            if args.output_type == 'video':
                for j in range(config.SAMPLE_PARAMS['n_samples']):
                    output_file = f"{output_dir}/{output_dataset_id}_layers[{j:0>3d}].mp4"
                    img.save_video(out['layers'][j], output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_layers"
                misc.create_dir(output_subdir)
                for j in range(config.SAMPLE_PARAMS['n_samples']):
                    img.save(out['layers'][j], [
                        f'{output_subdir}/{i:0>4d}[{j:0>3d}].png'
                        for i in dataset.view_idxs
                    ])


def test1():
    with torch.no_grad():
        # 1. Load dataset
        print("Load dataset: " + data_desc_path)
        dataset = SphericalViewSynDataset(data_desc_path, res=args.output_res,
                                          load_images=args.output_flags['perf'],
                                          load_depths=True, load_bins=True)
        data_loader = FastDataLoader(dataset, 1, shuffle=False, pin_memory=True)

        # 2. Load trained model
        netio.load(test_model_path, model)
        model.eval()

        # 3. Test on dataset
        print("Begin test, batch size is %d" % TEST_BATCH_SIZE)

        i = 0
        global_offset = 0
        chns = color.chns(config.COLOR)
        n = dataset.n_views
        total_pixels = n * dataset.view_res[0] * dataset.view_res[1]

        out = {}
        if args.output_flags['perf']:
            perf_times = torch.empty(n)
            perf = Perf(True, start=True)
        if args.output_flags['layers']:
            out['layers'] = torch.empty(total_pixels, config.SAMPLE_PARAMS['n_samples'], chns + 1,
                                        device=device.default())
        if args.output_flags['perf'] or args.output_flags['color']:
            out['color'] = torch.empty(total_pixels, chns, device=device.default())
        if args.output_flags['depth']:
            out['depth'] = torch.empty(total_pixels, device=device.default())

        for vi, _, rays_o, rays_d in data_loader:
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)
            rays_depth = dataset.patched_depths[vi].flatten() if dataset.load_depths else None
            rays_bins = dataset.patched_bins[vi].flatten(0, 2) if dataset.load_bins else None
            n_rays = rays_o.size(0)
            for offset in range(0, n_rays, TEST_MAX_RAYS):
                idx = slice(offset, min(offset + TEST_MAX_RAYS, n_rays))
                global_idx = slice(idx.start + global_offset, idx.stop + global_offset)
                ret = model(rays_o[idx], rays_d[idx],
                            rays_depth[idx] if rays_depth is not None else None,
                            rays_bins[idx] if rays_bins is not None else None,
                            ret_depth=args.output_flags['depth'],
                            debug=args.output_flags['layers'])
                if isinstance(ret, torch.Tensor):
                    ret = {'color': ret}
                if isinstance(ret, list):
                    ret = ret[-1]
                for key in out:
                    out[key][global_idx] = ret[key]
            if args.output_flags['perf']:
                perf_times[i] = perf.checkpoint()
            progress_bar(i, n, 'Inferring...')
            i += 1
            global_offset += n_rays

        # 4. Save results
        print('Saving results...')
        misc.create_dir(output_dir)

        for key in out:
            shape = [n] + list(dataset.view_res) + list(out[key].size()[1:])
            out[key] = out[key].view(shape)
        if 'color' in out:
            out['color'] = out['color'].permute(0, 3, 1, 2)
        if 'layers' in out:
            # n, y, x, samples, chns -> samples, n, chns, y, x
            out['layers'] = out['layers'].permute(3, 0, 4, 1, 2)

        if args.output_flags['perf']:
            perf_errors = torch.ones(n) * NaN
            perf_ssims = torch.ones(n) * NaN
            if dataset.view_images != None:
                for i in range(n):
                    perf_errors[i] = loss_mse(dataset.view_images[i], out['color'][i]).item()
                    perf_ssims[i] = ssim(dataset.view_images[i:i + 1],
                                         out['color'][i:i + 1]).item() * 100
            perf_mean_time = torch.mean(perf_times).item()
            perf_mean_error = torch.mean(perf_errors).item()
            perf_name = 'perf_%s_%.1fms_%.2e.csv' % (
                output_dataset_id, perf_mean_time, perf_mean_error)

            # Remove old performance reports
            for file in os.listdir(output_dir):
                if file.startswith(f'perf_{output_dataset_id}'):
                    os.remove(f"{output_dir}/{file}")

            # Save new performance reports
            with open(f"{output_dir}/{perf_name}", 'w') as fp:
                fp.write('View, Time, PSNR, SSIM\n')
                fp.writelines([
                    f'{dataset.view_idxs[i]}, {perf_times[i].item():.2f}, '
                    f'{img.mse2psnr(perf_errors[i].item()):.2f}, {perf_ssims[i].item():.2f}\n'
                    for i in range(n)
                ])

        if args.output_flags['color']:
            if args.output_type == 'video':
                output_file = f"{output_dir}/{output_dataset_id}_color.mp4"
                img.save_video(out['color'], output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_color"
                misc.create_dir(output_subdir)
                img.save(out['color'], [f'{output_subdir}/{i:0>4d}.png' for i in dataset.view_idxs])

        if args.output_flags['depth']:
            colorized_depths = img.colorize_depthmap(
                out['depth'], config.SAMPLE_PARAMS['depth_range'])
            if args.output_type == 'video':
                output_file = f"{output_dir}/{output_dataset_id}_depth.mp4"
                img.save_video(colorized_depths, output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_depth"
                misc.create_dir(output_subdir)
                img.save(colorized_depths, [
                    f'{output_subdir}/{i:0>4d}.png'
                    for i in dataset.view_idxs
                ])

        if args.output_flags['layers']:
            if args.output_type == 'video':
                for j in range(config.SAMPLE_PARAMS['n_samples']):
                    output_file = f"{output_dir}/{output_dataset_id}_layers[{j:0>3d}].mp4"
                    img.save_video(out['layers'][j], output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_layers"
                misc.create_dir(output_subdir)
                for j in range(config.SAMPLE_PARAMS['n_samples']):
                    img.save(out['layers'][j], [
                        f'{output_subdir}/{i:0>4d}[{j:0>3d}].png'
                        for i in dataset.view_idxs
                    ])


def test2():
    with torch.no_grad():
        # 1. Load dataset
        print("Load dataset: " + data_desc_path)
        dataset = SphericalViewSynDataset(data_desc_path, res=args.output_res,
                                          load_images=args.output_flags['perf'],
                                          load_depths=True)
        data_loader = FastDataLoader(dataset, 1, shuffle=False, pin_memory=True)

        # 2. Load trained model
        netio.load(test_model_path, model)
        model.set_depth_maps(dataset.rays_o, dataset.rays_d, dataset.view_depths)
        model.eval()

        # 3. Test on dataset
        print("Begin test, batch size is %d" % TEST_BATCH_SIZE)

        i = 0
        global_offset = 0
        chns = color.chns(config.COLOR)
        n = dataset.n_views
        total_pixels = n * dataset.view_res[0] * dataset.view_res[1]

        out = {}
        if args.output_flags['perf']:
            perf_times = torch.empty(n)
            perf = Perf(True, start=True)
        if args.output_flags['layers']:
            out['layers'] = torch.empty(total_pixels, config.SAMPLE_PARAMS['n_samples'], chns + 1,
                                        device=device.default())
        if args.output_flags['perf'] or args.output_flags['color']:
            out['color'] = torch.empty(total_pixels, chns, device=device.default())
        if args.output_flags['depth']:
            out['depth'] = torch.empty(total_pixels, device=device.default())

        for vi, _, rays_o, rays_d in data_loader:
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)
            rays_weights = model.bin_weights[vi].flatten(0, 2)
            n_rays = rays_o.size(0)
            for offset in range(0, n_rays, TEST_MAX_RAYS):
                idx = slice(offset, min(offset + TEST_MAX_RAYS, n_rays))
                global_idx = slice(idx.start + global_offset, idx.stop + global_offset)
                ret = model(rays_o[idx], rays_d[idx], rays_weights[idx],
                            ret_depth=args.output_flags['depth'],
                            debug=args.output_flags['layers'])
                if isinstance(ret, torch.Tensor):
                    ret = {'color': ret}
                if isinstance(ret, list):
                    ret = ret[-1]
                for key in out:
                    out[key][global_idx] = ret[key]
            if args.output_flags['perf']:
                perf_times[i] = perf.checkpoint()
            progress_bar(i, n, 'Inferring...')
            i += 1
            global_offset += n_rays

        # 4. Save results
        print('Saving results...')
        misc.create_dir(output_dir)

        for key in out:
            shape = [n] + list(dataset.view_res) + list(out[key].size()[1:])
            out[key] = out[key].view(shape)
        if 'color' in out:
            out['color'] = out['color'].permute(0, 3, 1, 2)
        if 'layers' in out:
            # n, y, x, samples, chns -> samples, n, chns, y, x
            out['layers'] = out['layers'].permute(3, 0, 4, 1, 2)

        if args.output_flags['perf']:
            perf_errors = torch.ones(n) * NaN
            perf_ssims = torch.ones(n) * NaN
            if dataset.view_images != None:
                for i in range(n):
                    perf_errors[i] = loss_mse(dataset.view_images[i], out['color'][i]).item()
                    perf_ssims[i] = ssim(dataset.view_images[i:i + 1],
                                         out['color'][i:i + 1]).item() * 100
            perf_mean_time = torch.mean(perf_times).item()
            perf_mean_error = torch.mean(perf_errors).item()
            perf_name = 'perf_%s_%.1fms_%.2e.csv' % (
                output_dataset_id, perf_mean_time, perf_mean_error)

            # Remove old performance reports
            for file in os.listdir(output_dir):
                if file.startswith(f'perf_{output_dataset_id}'):
                    os.remove(f"{output_dir}/{file}")

            # Save new performance reports
            with open(f"{output_dir}/{perf_name}", 'w') as fp:
                fp.write('View, Time, PSNR, SSIM\n')
                fp.writelines([
                    f'{dataset.view_idxs[i]}, {perf_times[i].item():.2f}, '
                    f'{img.mse2psnr(perf_errors[i].item()):.2f}, {perf_ssims[i].item():.2f}\n'
                    for i in range(n)
                ])

        if args.output_flags['color']:
            if args.output_type == 'video':
                output_file = f"{output_dir}/{output_dataset_id}_color.mp4"
                img.save_video(out['color'], output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_color"
                misc.create_dir(output_subdir)
                img.save(out['color'], [f'{output_subdir}/{i:0>4d}.png' for i in dataset.view_idxs])

        if args.output_flags['depth']:
            colorized_depths = img.colorize_depthmap(
                out['depth'], config.SAMPLE_PARAMS['depth_range'])
            if args.output_type == 'video':
                output_file = f"{output_dir}/{output_dataset_id}_depth.mp4"
                img.save_video(colorized_depths, output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_depth"
                misc.create_dir(output_subdir)
                img.save(colorized_depths, [
                    f'{output_subdir}/{i:0>4d}.png'
                    for i in dataset.view_idxs
                ])

        if args.output_flags['layers']:
            if args.output_type == 'video':
                for j in range(config.SAMPLE_PARAMS['n_samples']):
                    output_file = f"{output_dir}/{output_dataset_id}_layers[{j:0>3d}].mp4"
                    img.save_video(out['layers'][j], output_file, 30)
            else:
                output_subdir = f"{output_dir}/{output_dataset_id}_layers"
                misc.create_dir(output_subdir)
                for j in range(config.SAMPLE_PARAMS['n_samples']):
                    img.save(out['layers'][j], [
                        f'{output_subdir}/{i:0>4d}[{j:0>3d}].png'
                        for i in dataset.view_idxs
                    ])


if __name__ == "__main__":
    if args.test:
        if is_dnerf:
            test1()
        elif is_cnerf:
            test2()
        else:
            test()
    else:
        train()
