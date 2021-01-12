from __future__ import print_function

import sys
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
from math import log10
from my.progress_bar import progress_bar
from my import color_mode


class Net(torch.nn.Module):
    def __init__(self, color, base_filter):
        super(Net, self).__init__()
        self.color = color
        if color == color_mode.GRAY:
            self.layers = torch.nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=1, padding=2, bias=True),
                #nn.PixelShuffle(upscale_factor)
            )
        else:
            self.net_1 = torch.nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=1, padding=2, bias=True),
                #nn.PixelShuffle(upscale_factor)
            )
            self.net_2 = torch.nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=1, padding=2, bias=True),
                #nn.PixelShuffle(upscale_factor)
            )
            self.net_3 = torch.nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=base_filter // 2, out_channels=1, kernel_size=5, stride=1, padding=2, bias=True),
                #nn.PixelShuffle(upscale_factor)
            )

    def forward(self, x):
        if self.color == color_mode.GRAY:
            out = self.layers(x)
        else:
            out = torch.cat([
                self.net_1(x[:, 0:1]),
                self.net_2(x[:, 1:2]),
                self.net_3(x[:, 2:3])
            ], dim=1)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Solver(object):
    def __init__(self, config, training_loader, testing_loader, writer=None):
        super(Solver, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.writer = writer
        self.color = config.color

    def build_model(self):
        self.model = Net(color=self.color, base_filter=64).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self):
        model_out_path = "model_path.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self, epoch, iters, channels = None):
        self.model.train()
        train_loss = 0
        for batch_num, (_, data, target) in enumerate(self.training_loader):
            if channels:
                data = data[..., channels, :, :]
                target = target[..., channels, :, :]
            data =data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            sys.stdout.write('Epoch %d: ' % epoch)
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))
            if self.writer:
                self.writer.add_scalar("loss", loss, iters)
                if iters % 100 == 0:
                    output_vs_gt = torch.stack([out, target], 1) \
                        .flatten(0, 1).detach()
                    self.writer.add_image(
                        "Output_vs_gt",
                        torchvision.utils.make_grid(output_vs_gt, nrow=2).cpu().numpy(),
                        iters)
            iters += 1

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        return iters