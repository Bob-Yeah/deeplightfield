import torch
from .ssim import *
from .perc_loss import *

device=torch.device("cuda:2")

l1loss = torch.nn.L1Loss()
perc_loss = VGGPerceptualLoss().to(device)

##### LOSS #####


def calImageGradients(images):
    # x is a 4-D tensor
    dx = images[:, :, 1:, :] - images[:, :, :-1, :]
    dy = images[:, :, :, 1:] - images[:, :, :, :-1]
    return dx, dy


def loss_new(generated, gt):
    mse_loss = torch.nn.MSELoss()
    rmse_intensity = mse_loss(generated, gt)
    psnr_intensity = torch.log10(rmse_intensity)
    # print("psnr:",psnr_intensity)
    # ssim_intensity = ssim(generated, gt)
    labels_dx, labels_dy = calImageGradients(gt)
    # print("generated:",generated.shape)
    preds_dx, preds_dy = calImageGradients(generated)
    rmse_grad_x, rmse_grad_y = mse_loss(
        labels_dx, preds_dx), mse_loss(labels_dy, preds_dy)
    psnr_grad_x, psnr_grad_y = torch.log10(
        rmse_grad_x), torch.log10(rmse_grad_y)
    # print("psnr x&y:",psnr_grad_x," ",psnr_grad_y)
    p_loss = perc_loss(generated, gt)
    # print("-psnr:",-psnr_intensity,",0.5*(psnr_grad_x + psnr_grad_y):",0.5*(psnr_grad_x + psnr_grad_y),",perc_loss:",p_loss)
    total_loss = psnr_intensity + 0.5*(psnr_grad_x + psnr_grad_y) + p_loss
    # total_loss = rmse_intensity + 0.5*(rmse_grad_x + rmse_grad_y) # + p_loss
    return total_loss


def loss_without_perc(generated, gt):
    mse_loss = torch.nn.MSELoss()
    rmse_intensity = mse_loss(generated, gt)
    psnr_intensity = torch.log10(rmse_intensity)
    # print("psnr:",psnr_intensity)
    # ssim_intensity = ssim(generated, gt)
    labels_dx, labels_dy = calImageGradients(gt)
    # print("generated:",generated.shape)
    preds_dx, preds_dy = calImageGradients(generated)
    rmse_grad_x, rmse_grad_y = mse_loss(
        labels_dx, preds_dx), mse_loss(labels_dy, preds_dy)
    psnr_grad_x, psnr_grad_y = torch.log10(
        rmse_grad_x), torch.log10(rmse_grad_y)
    # print("psnr x&y:",psnr_grad_x," ",psnr_grad_y)
    # print("-psnr:",-psnr_intensity,",0.5*(psnr_grad_x + psnr_grad_y):",0.5*(psnr_grad_x + psnr_grad_y),",perc_loss:",p_loss)
    total_loss = psnr_intensity + 0.5*(psnr_grad_x + psnr_grad_y)
    # total_loss = rmse_intensity + 0.5*(rmse_grad_x + rmse_grad_y) # + p_loss
    return total_loss
##### LOSS #####


class ReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, generated, gt):
        rmse_intensity = torch.nn.functional.mse_loss(generated, gt)
        psnr_intensity = torch.log10(rmse_intensity)
        labels_dx, labels_dy = calImageGradients(gt)
        preds_dx, preds_dy = calImageGradients(generated)
        rmse_grad_x, rmse_grad_y = torch.nn.functional.mse_loss(
            labels_dx, preds_dx), torch.nn.functional.mse_loss(labels_dy, preds_dy)
        psnr_grad_x, psnr_grad_y = torch.log10(
            rmse_grad_x), torch.log10(rmse_grad_y)
        total_loss = psnr_intensity + 0.5*(psnr_grad_x + psnr_grad_y)
        return total_loss


class PerceptionReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptionReconstructionLoss, self).__init__()

    def forward(self, generated, gt):
        rmse_intensity = torch.nn.functional.mse_loss(generated, gt)
        psnr_intensity = torch.log10(rmse_intensity)
        labels_dx, labels_dy = calImageGradients(gt)
        preds_dx, preds_dy = calImageGradients(generated)
        rmse_grad_x = torch.nn.functional.mse_loss(labels_dx, preds_dx)
        rmse_grad_y = torch.nn.functional.mse_loss(labels_dy, preds_dy)
        psnr_grad_x = torch.log10(rmse_grad_x)
        psnr_grad_y = torch.log10(rmse_grad_y)
        p_loss = perc_loss(generated, gt)
        total_loss = psnr_intensity + 0.5 * (psnr_grad_x + psnr_grad_y) + p_loss
        return total_loss
