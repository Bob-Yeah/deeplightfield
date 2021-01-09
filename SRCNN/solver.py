from __future__ import print_function

from math import log10
import sys

import torch
import torch.backends.cudnn as cudnn
import torchvision

from .model import Net
from ..my.progress_bar import progress_bar


class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader, writer=None):
        super(SRCNNTrainer, self).__init__()
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

    def build_model(self):
        self.model = Net(num_channels=1, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
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

    def train(self, epoch, iters):
        self.model.train()
        train_loss = 0
        for batch_num, (_, data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
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

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save_model()
