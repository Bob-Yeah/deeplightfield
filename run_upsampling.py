from __future__ import print_function

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from tensorboardX.writer import SummaryWriter

sys.path.append(os.path.abspath(sys.path[0] + '/../'))
__package__ = "deeplightfield"

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--device', type=int, default=3,
                    help='Which CUDA device to use.')
parser.add_argument('--batchSize', type=int, default=1,
                    help='training batch size')
parser.add_argument('--testBatchSize', type=int,
                    default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset directory')
parser.add_argument('--test', type=str, help='path of model to test')
parser.add_argument('--testOutPatt', type=str, help='test output path pattern')

# model configuration
parser.add_argument('--upscale_factor', '-uf', type=int,
                    default=2, help="super resolution upscale factor")
#parser.add_argument('--model', '-m', type=str, default='srgan', help='choose which model is going to use')

args = parser.parse_args()

# Select device
torch.cuda.set_device(args.device)
print("Set CUDA:%d as current device." % torch.cuda.current_device())

from .my import util
from .my import netio
from .my import device
from .SRGAN.solver import SRGANTrainer as Solver
from .data.upsampling import UpsamplingDataset
from .data.loader import FastDataLoader

os.chdir(args.dataset)
print('Change working directory to ' + os.getcwd())
run_dir = 'run/'


def train():
    util.CreateDirIfNeed(run_dir)
    train_set = UpsamplingDataset('.', 'out_view_%04d.png',
                                  'gt_view_%04d.png', gray=True)
    training_data_loader = FastDataLoader(dataset=train_set,
                                          batch_size=args.batchSize,
                                          shuffle=True,
                                          drop_last=False)
    trainer = Solver(args, training_data_loader, training_data_loader,
                     SummaryWriter(run_dir))
    trainer.build_model()
    # ===
    for epoch in range(1, 20 + 1):
        trainer.pretrain()
        print("{}/{} pretrained".format(epoch, trainer.epoch_pretrain))
    # ===
    iters = 0
    for epoch in range(1, args.nEpochs + 1):
        print("\n===> Epoch {} starts:".format(epoch))
        iters = trainer.train(epoch, iters)
    netio.SaveNet(run_dir + 'model-epoch_%d.pth' % args.nEpochs, trainer.netG)


def test():
    util.CreateDirIfNeed(os.path.dirname(args.testOutPatt))
    train_set = UpsamplingDataset('.', 'out_view_%04d.png', None, gray=True)
    training_data_loader = FastDataLoader(dataset=train_set,
                                          batch_size=args.testBatchSize,
                                          shuffle=False,
                                          drop_last=False)
    trainer = Solver(args, training_data_loader, training_data_loader,
                     SummaryWriter(run_dir))
    trainer.build_model()
    netio.LoadNet(args.test, trainer.netG)
    for idx, input, _ in training_data_loader:
        output = trainer.netG(input)
        util.WriteImageTensor(output, args.testOutPatt % idx)


def main():
    if (args.test):
        test()
    else:
        train()


if __name__ == '__main__':
    main()
