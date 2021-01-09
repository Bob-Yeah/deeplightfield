import math
import sys
import os
import argparse
import torch
import torch.optim
import torchvision
from tensorboardX import SummaryWriter
from torch import nn

sys.path.append(os.path.abspath(sys.path[0] + '/../'))
__package__ = "deeplightfield"

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=3,
                    help='Which CUDA device to use.')
parser.add_argument('--config', type=str,
                    help='Net config files')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset description file')
parser.add_argument('--test', type=str,
                    help='Test net file')
parser.add_argument('--test-samples', type=int,
                    help='Samples used for test')
parser.add_argument('--output-gt', action='store_true',
                    help='Output ground truth images if exist')
parser.add_argument('--output-alongside', action='store_true',
                    help='Output generated image alongside ground truth image')
parser.add_argument('--output-video', action='store_true',
                    help='Output test results as video')
opt = parser.parse_args()


# Select device
torch.cuda.set_device(opt.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())

from .my import netio
from .my import util
from .my import device
from .my import loss
from .my.simple_perf import SimplePerf
from .data.spherical_view_syn import *
from .data.loader import FastDataLoader
from .msl_net import MslNet
from .spher_net import SpherNet
from .configs.spherical_view_syn import SphericalViewSynConfig


config = SphericalViewSynConfig()

# Toggles
ROT_ONLY = False
EVAL_TIME_PERFORMANCE = False
# ========
#ROT_ONLY = True
#EVAL_TIME_PERFORMANCE = True

# Train
BATCH_SIZE = 4096
EPOCH_RANGE = range(0, 500)
SAVE_INTERVAL = 20

# Test
TEST_BATCH_SIZE = 1
TEST_MAX_RAYS = 32768

# Paths
data_desc_path = opt.dataset
data_desc_name = os.path.split(data_desc_path)[1]
if opt.test:
    test_net_path = opt.test
    test_net_name = os.path.splitext(os.path.basename(test_net_path))[0]
    run_dir = os.path.dirname(test_net_path) + '/'
    run_id = os.path.basename(run_dir[:-1])
    output_dir = run_dir + 'output/%s/%s/' % (test_net_name, data_desc_name)
    config.from_id(run_id)
    train_mode = False
    if opt.test_samples:
        config.SAMPLE_PARAMS['n_samples'] = opt.test_samples
        output_dir = run_dir + 'output/%s/%s_s%d/' % \
            (test_net_name, data_desc_name, opt.test_samples)
else:
    if opt.config:
        config.load(opt.config)
    data_dir = os.path.dirname(data_desc_path) + '/'
    run_id = config.to_id()
    run_dir = data_dir + run_id + '/'
    log_dir = run_dir + 'log/'
    output_dir = None
    train_mode = True

config.print()
print("dataset: ", data_desc_path)
print("train_mode: ", train_mode)
print("run_dir: ", run_dir)
if not train_mode:
    print("output_dir", output_dir)

config.SAMPLE_PARAMS['perturb_sample'] = \
    config.SAMPLE_PARAMS['perturb_sample'] and train_mode

NETS = {
    'msl': lambda: MslNet(
        fc_params=config.FC_PARAMS,
        sampler_params=(config.SAMPLE_PARAMS.update(
            {'spherical': True}), config.SAMPLE_PARAMS)[1],
        gray=config.GRAY,
        encode_to_dim=config.N_ENCODE_DIM),
    'nerf': lambda: MslNet(
        fc_params=config.FC_PARAMS,
        sampler_params=(config.SAMPLE_PARAMS.update(
            {'spherical': False}), config.SAMPLE_PARAMS)[1],
        gray=config.GRAY,
        encode_to_dim=config.N_ENCODE_DIM),
    'spher': lambda: SpherNet(
        fc_params=config.FC_PARAMS,
        gray=config.GRAY,
        translation=not ROT_ONLY,
        encode_to_dim=config.N_ENCODE_DIM)
}

LOSSES = {
    'mse': lambda: nn.MSELoss(),
    'mse_grad': lambda: loss.CombinedLoss(
        [nn.MSELoss(), loss.GradLoss()], [1.0, 0.5])
}

# Initialize model
model = NETS[config.NET_TYPE]().to(device.GetDevice())
loss_mse = nn.MSELoss().to(device.GetDevice())
loss_grad = loss.GradLoss().to(device.GetDevice())


def train_loop(data_loader, optimizer, loss, perf, writer, epoch, iters):
    sub_iters = 0
    iters_in_epoch = len(data_loader)
    for _, gt, rays_o, rays_d in data_loader:
        patch = (len(gt.size()) == 4)
        gt = gt.to(device.GetDevice())
        rays_o = rays_o.to(device.GetDevice())
        rays_d = rays_d.to(device.GetDevice())
        perf.Checkpoint("Load")

        out = model(rays_o, rays_d)
        perf.Checkpoint("Forward")

        optimizer.zero_grad()
        loss_mse_value = loss_mse(out, gt)
        loss_grad_value = loss_grad(out, gt) if patch else None
        loss_value = loss_mse_value  # + 0.5 * loss_grad_value if patch \
        # else loss_mse_value
        perf.Checkpoint("Compute loss")

        loss_value.backward()
        perf.Checkpoint("Backward")

        optimizer.step()
        perf.Checkpoint("Update")

        if patch:
            print("Epoch: %d, Iter: %d(%d/%d), Loss MSE: %f, Loss Grad: %f" %
                  (epoch, iters, sub_iters, iters_in_epoch,
                   loss_mse_value.item(), loss_grad_value.item()))
        else:
            print("Epoch: %d, Iter: %d(%d/%d), Loss MSE: %f" %
                  (epoch, iters, sub_iters, iters_in_epoch, loss_mse_value.item()))

        # Write tensorboard logs.
        writer.add_scalar("loss mse", loss_mse_value, iters)
        if patch:
            writer.add_scalar("loss grad", loss_grad_value, iters)
        if patch and iters % 100 == 0:
            output_vs_gt = torch.cat([out[0:4], gt[0:4]], 0).detach()
            writer.add_image("Output_vs_gt", torchvision.utils.make_grid(
                output_vs_gt, nrow=4).cpu().numpy(), iters)

        iters += 1
        sub_iters += 1
    return iters


def train():
    # 1. Initialize data loader
    print("Load dataset: " + data_desc_path)
    train_dataset = SphericalViewSynDataset(data_desc_path, gray=config.GRAY)
    train_dataset.set_patch_size(1)
    train_data_loader = FastDataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=True)

    # 2. Initialize components
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss = 0#LOSSES[config.LOSS]().to(device.GetDevice())

    if EPOCH_RANGE.start > 0:
        iters = netio.LoadNet('%smodel-epoch_%d.pth' % (run_dir, EPOCH_RANGE.start),
                              model, solver=optimizer)
    else:
        iters = 0
    epoch = None

    # 3. Train
    model.train()

    util.CreateDirIfNeed(run_dir)
    util.CreateDirIfNeed(log_dir)

    perf = SimplePerf(EVAL_TIME_PERFORMANCE, start=True)
    perf_epoch = SimplePerf(True, start=True)
    writer = SummaryWriter(log_dir)

    print("Begin training...")
    for epoch in EPOCH_RANGE:
        perf_epoch.Checkpoint("Epoch")
        iters = train_loop(train_data_loader, optimizer, loss,
                           perf, writer, epoch, iters)
        # Save checkpoint
        if ((epoch + 1) % SAVE_INTERVAL == 0):
            netio.SaveNet('%smodel-epoch_%d.pth' % (run_dir, epoch + 1), model,
                          solver=optimizer, iters=iters)
    print("Train finished")


def test():
    with torch.no_grad():
        # 1. Load train dataset
        print("Load dataset: " + data_desc_path)
        test_dataset = SphericalViewSynDataset(data_desc_path,
                                               load_images=opt.output_gt or opt.output_alongside,
                                               gray=config.GRAY)
        test_data_loader = FastDataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

        # 2. Load trained model
        netio.LoadNet(test_net_path, model)

        # 3. Test on train dataset
        print("Begin test on train dataset, batch size is %d" % TEST_BATCH_SIZE)
        util.CreateDirIfNeed(output_dir)

        perf = SimplePerf(True, start=True)
        i = 0
        n = test_dataset.n_views
        chns = 1 if config.GRAY else 3
        out_view_images = torch.empty(n, chns, test_dataset.view_res[0],
                                      test_dataset.view_res[1],
                                      device=device.GetDevice())
        for view_idxs, _, rays_o, rays_d in test_data_loader:
            perf.Checkpoint("%d - Load" % i)
            rays_o = rays_o.to(device.GetDevice()).view(-1, 3)
            rays_d = rays_d.to(device.GetDevice()).view(-1, 3)
            n_rays = rays_o.size(0)
            chunk_size = min(n_rays, TEST_MAX_RAYS)
            out_pixels = torch.empty(n_rays, chns, device=device.GetDevice())
            for offset in range(0, n_rays, chunk_size):
                idx = slice(offset, offset + chunk_size)
                out_pixels[idx] = model(rays_o[idx], rays_d[idx])
            out_view_images[view_idxs] = out_pixels.view(
                TEST_BATCH_SIZE, test_dataset.view_res[0],
                test_dataset.view_res[1], -1).permute(0, 3, 1, 2)
            perf.Checkpoint("%d - Infer" % i)
            i += 1

        # 4. Save results
        if opt.output_video:
            util.generate_video(out_view_images, output_dir +
                                'out.mp4', 24, 3, True)
        else:
            gt_paths = [
                '%sgt_view_%04d.png' % (output_dir, i) for i in range(n)
            ]
            out_paths = [
                '%sout_view_%04d.png' % (output_dir, i) for i in range(n)
            ]
            if test_dataset.load_images:
                if opt.output_alongside:
                    util.WriteImageTensor(
                        torch.cat([
                            test_dataset.view_images,
                            out_view_images
                        ], 3), out_paths)
                else:
                    util.WriteImageTensor(out_view_images, out_paths)
                    util.WriteImageTensor(test_dataset.view_images, gt_paths)
            else:
                util.WriteImageTensor(out_view_images, out_paths)


if __name__ == "__main__":
    if train_mode:
        train()
    else:
        test()
