import sys
sys.path.append('/e/dengnc')
__package__ = "deep_view_syn"

import os
import torch
import torch.optim
import torchvision
from tensorboardX import SummaryWriter
from utils.loss import PerceptionReconstructionLoss
from utils import netio
from utils import misc
from utils import device
from utils import img
from utils.perf import Perf
from data.lf_syn import LightFieldSynDataset
from nets.trans_unet import TransUnet


torch.cuda.set_device(2)
print("Set CUDA:%d as current device." % torch.cuda.current_device())

DATA_DIR = os.path.dirname(__file__) + '/data/lf_syn_2020.12.23'
TRAIN_DATA_DESC_FILE = DATA_DIR + '/train.json'
OUTPUT_DIR = DATA_DIR + '/output_bat2'
RUN_DIR = DATA_DIR + '/run_bat2'
BATCH_SIZE = 8
TEST_BATCH_SIZE = 10
NUM_EPOCH = 1000
MODE = "Silence"  # "Perf"
EPOCH_BEGIN = 600


def train():
    # 1. Initialize data loader
    print("Load dataset: " + TRAIN_DATA_DESC_FILE)
    train_dataset = LightFieldSynDataset(TRAIN_DATA_DESC_FILE)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        drop_last=False)
    print(len(train_data_loader))

    # 2. Initialize components
    model = TransUnet(cam_params=train_dataset.cam_params,
                      view_images=train_dataset.sparse_view_images,
                      view_depths=train_dataset.sparse_view_depths,
                      view_positions=train_dataset.sparse_view_positions,
                      diopter_of_layers=train_dataset.diopter_of_layers).to(device.default())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = PerceptionReconstructionLoss()

    if EPOCH_BEGIN > 0:
        netio.load('%s/model-epoch_%d.pth' % (RUN_DIR, EPOCH_BEGIN), model,
                      solver=optimizer)

    # 3. Train
    model.train()
    epoch = EPOCH_BEGIN
    iters = EPOCH_BEGIN * len(train_data_loader) * BATCH_SIZE

    misc.create_dir(RUN_DIR)

    perf = Perf(enable=(MODE == "Perf"), start=True)
    writer = SummaryWriter(RUN_DIR)

    print("Begin training...")
    for epoch in range(EPOCH_BEGIN, NUM_EPOCH):
        for _, view_images, _, view_positions in train_data_loader:

            view_images = view_images.to(device.default())

            perf.checkpoint("Load")

            out_view_images = model(view_positions)

            perf.checkpoint("Forward")

            optimizer.zero_grad()
            loss_value = loss(out_view_images, view_images)

            perf.checkpoint("Compute loss")

            loss_value.backward()

            perf.checkpoint("Backward")

            optimizer.step()

            perf.checkpoint("Update")

            print("Epoch: ", epoch, ", Iter: ", iters,
                  ", Loss: ", loss_value.item())

            iters = iters + BATCH_SIZE

            # Write tensorboard logs.
            writer.add_scalar("loss", loss_value, iters)
            if iters % len(train_data_loader) == 0:
                output_vs_gt = torch.cat([out_view_images, view_images], dim=0)
                writer.add_image("Output_vs_gt", torchvision.utils.make_grid(
                    output_vs_gt, scale_each=True, normalize=False)
                    .cpu().detach().numpy(), iters)

        # Save checkpoint
        if ((epoch + 1) % 50 == 0):
            netio.save('%s/model-epoch_%d.pth' % (RUN_DIR, epoch + 1), model, iters)

    print("Train finished")


def test(net_file: str):
    # 1. Load train dataset
    print("Load dataset: " + TRAIN_DATA_DESC_FILE)
    train_dataset = LightFieldSynDataset(TRAIN_DATA_DESC_FILE)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TEST_BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        drop_last=False)

    # 2. Load trained model
    model = TransUnet(cam_params=train_dataset.cam_params,
                      view_images=train_dataset.sparse_view_images,
                      view_depths=train_dataset.sparse_view_depths,
                      view_positions=train_dataset.sparse_view_positions,
                      diopter_of_layers=train_dataset.diopter_of_layers).to(device.default())
    netio.load(net_file, model)

    # 3. Test on train dataset
    print("Begin test on train dataset...")
    misc.create_dir(OUTPUT_DIR)
    for view_idxs, view_images, _, view_positions in train_data_loader:
        out_view_images = model(view_positions)
        img.save(view_images,
                 '%s/gt_view%02d.png' % (OUTPUT_DIR, i) for i in view_idxs)
        img.save(out_view_images,
                 '%s/out_view%02d.png' % (OUTPUT_DIR, i) for i in view_idxs)


if __name__ == "__main__":
    # train()
    test(RUN_DIR + '/model-epoch_1000.pth')
