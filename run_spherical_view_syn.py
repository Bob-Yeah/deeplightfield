import sys
sys.path.append('/e/dengnc')
__package__ = "deeplightfield"

import argparse
import torch
import torch.optim
import torchvision
from typing import List, Tuple
from tensorboardX import SummaryWriter
from torch import nn
from .my import netio
from .my import util
from .my import device
from .my.simple_perf import SimplePerf
from .loss.loss import PerceptionReconstructionLoss
from .data.spherical_view_syn import SphericalViewSynDataset
from .msl_net import MslNet
from .spher_net import SpherNet


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=3,
                    help='Which CUDA device to use.')
opt = parser.parse_args()


# Select device
torch.cuda.set_device(opt.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())

# Toggles
GRAY = False
ROT_ONLY = False
TRAIN_MODE = True
EVAL_TIME_PERFORMANCE = False
RAY_AS_ITEM = True
# ========
#GRAY = True
ROT_ONLY = True
TRAIN_MODE = False
#EVAL_TIME_PERFORMANCE = True
#RAY_AS_ITEM = False

# Net parameters
DEPTH_RANGE = (1, 10)
N_DEPTH_LAYERS = 10
N_ENCODE_DIM = 10
FC_PARAMS = {
    'nf': 128,
    'n_layers': 6,
    'skips': [4]
}

# Train
BATCH_SIZE = 2048 if RAY_AS_ITEM else 4
EPOCH_RANGE = range(0, 500)
SAVE_INTERVAL = 20

# Paths
DATA_DIR = sys.path[0] + '/data/sp_view_syn_2020.12.26_rotonly/'
RUN_ID = '%s_ray_b%d_encode%d_fc%dx%d%s' % ('gray' if GRAY else 'rgb',
                                            BATCH_SIZE,
                                            N_ENCODE_DIM,
                                            FC_PARAMS['nf'],
                                            FC_PARAMS['n_layers'],
                                            '_skip_%d' % FC_PARAMS['skips'][0] if len(FC_PARAMS['skips']) > 0 else '')
TRAIN_DATA_DESC_FILE = DATA_DIR + 'train.json'
RUN_DIR = DATA_DIR + RUN_ID + '/'
OUTPUT_DIR = RUN_DIR + 'output/'
LOG_DIR = RUN_DIR + 'log/'


# Test
TEST_NET_NAME = 'model-epoch_100'
TEST_BATCH_SIZE = 5


def train():
    # 1. Initialize data loader
    print("Load dataset: " + TRAIN_DATA_DESC_FILE)
    train_dataset = SphericalViewSynDataset(
        TRAIN_DATA_DESC_FILE, gray=GRAY, ray_as_item=RAY_AS_ITEM)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        drop_last=False)
    print('Data loaded. %d iters per epoch.' % len(train_data_loader))

    # 2. Initialize components
    if ROT_ONLY:
        model = SpherNet(cam_params=train_dataset.cam_params,
                         fc_params=FC_PARAMS,
                         out_res=train_dataset.view_res,
                         gray=GRAY,
                         encode_to_dim=N_ENCODE_DIM).to(device.GetDevice())
    else:
        model = MslNet(cam_params=train_dataset.cam_params,
                       sphere_layers=util.GetDepthLayers(
                           DEPTH_RANGE, N_DEPTH_LAYERS),
                       out_res=train_dataset.view_res,
                       gray=GRAY).to(device.GetDevice())
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss = nn.MSELoss()

    if EPOCH_RANGE.start > 0:
        netio.LoadNet('%smodel-epoch_%d.pth' % (RUN_DIR, EPOCH_RANGE.start),
                      model, solver=optimizer)

    # 3. Train
    model.train()
    epoch = None
    iters = EPOCH_RANGE.start * len(train_data_loader)

    util.CreateDirIfNeed(RUN_DIR)
    util.CreateDirIfNeed(LOG_DIR)

    perf = SimplePerf(EVAL_TIME_PERFORMANCE, start=True)
    perf_epoch = SimplePerf(True, start=True)
    writer = SummaryWriter(LOG_DIR)

    print("Begin training...")
    for epoch in EPOCH_RANGE:
        for _, gt, ray_positions, ray_directions in train_data_loader:

            gt = gt.to(device.GetDevice())
            ray_positions = ray_positions.to(device.GetDevice())
            ray_directions = ray_directions.to(device.GetDevice())

            perf.Checkpoint("Load")

            out = model(ray_positions, ray_directions)

            perf.Checkpoint("Forward")

            optimizer.zero_grad()
            loss_value = loss(out, gt)

            perf.Checkpoint("Compute loss")

            loss_value.backward()

            perf.Checkpoint("Backward")

            optimizer.step()

            perf.Checkpoint("Update")

            print("Epoch: ", epoch, ", Iter: ", iters,
                  ", Loss: ", loss_value.item())

            # Write tensorboard logs.
            writer.add_scalar("loss", loss_value, iters)
            if not RAY_AS_ITEM and iters % 100 == 0:
                output_vs_gt = torch.cat([out, gt], dim=0)
                writer.add_image("Output_vs_gt", torchvision.utils.make_grid(
                    output_vs_gt, scale_each=True, normalize=False)
                    .cpu().detach().numpy(), iters)

            iters += 1

        perf_epoch.Checkpoint("Epoch")
        # Save checkpoint
        if ((epoch + 1) % SAVE_INTERVAL == 0):
            netio.SaveNet('%smodel-epoch_%d.pth' % (RUN_DIR, epoch + 1), model,
                          solver=optimizer)

    print("Train finished")


def test(net_file: str):
    # 1. Load train dataset
    print("Load dataset: " + TRAIN_DATA_DESC_FILE)
    train_dataset = SphericalViewSynDataset(TRAIN_DATA_DESC_FILE,
                                            load_images=True, gray=GRAY)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TEST_BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        drop_last=False)

    # 2. Load trained model
    if ROT_ONLY:
        model = SpherNet(cam_params=train_dataset.cam_params,
                         fc_params=FC_PARAMS,
                         out_res=train_dataset.view_res,
                         gray=GRAY,
                         encode_to_dim=N_ENCODE_DIM).to(device.GetDevice())
    else:
        model = MslNet(cam_params=train_dataset.cam_params,
                       sphere_layers=_GetSphereLayers(
                           DEPTH_RANGE, N_DEPTH_LAYERS),
                       out_res=train_dataset.view_res,
                       gray=GRAY).to(device.GetDevice())
    netio.LoadNet(net_file, model)

    # 3. Test on train dataset
    print("Begin test on train dataset, batch size is %d" % TEST_BATCH_SIZE)
    util.CreateDirIfNeed(OUTPUT_DIR)
    util.CreateDirIfNeed(OUTPUT_DIR + TEST_NET_NAME)
    perf = SimplePerf(True, start=True)
    i = 0
    for view_idxs, view_images, ray_positions, ray_directions in train_data_loader:
        ray_positions = ray_positions.to(device.GetDevice())
        ray_directions = ray_directions.to(device.GetDevice())
        perf.Checkpoint("%d - Load" % i)
        out_view_images = model(ray_positions, ray_directions)
        perf.Checkpoint("%d - Infer" % i)
        util.WriteImageTensor(
            view_images,
            ['%s%s/gt_view_%04d.png' % (OUTPUT_DIR, TEST_NET_NAME, i) for i in view_idxs])
        util.WriteImageTensor(
            out_view_images,
            ['%s%s/out_view_%04d.png' % (OUTPUT_DIR, TEST_NET_NAME, i) for i in view_idxs])
        perf.Checkpoint("%d - Write" % i)
        i += 1


if __name__ == "__main__":
    if TRAIN_MODE:
        train()
    else:
        test(RUN_DIR + TEST_NET_NAME + '.pth')
