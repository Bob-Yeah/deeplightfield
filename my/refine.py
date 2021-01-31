import cv2
import torch
import numpy as np
import torch.nn.functional as nn_f
from . import view
from . import util


class GuideRefinement(object):

    def __init__(self, guides_image, guides_view: view.Trans,
                 guides_cam: view.CameraParam, net) -> None:
        rays_o, rays_d = guides_cam.get_global_rays(guides_view, flatten=True)
        guides_inferred = torch.stack([
            net(rays_o[i], rays_d[i]).view(
                guides_cam.res[0], guides_cam.res[1], -1).permute(2, 0, 1)
            for i in range(guides_image.size(0))
        ], 0)
        self.guides_diff = (guides_image - guides_inferred) / \
            (guides_inferred + 1e-5)
        self.guides_view = guides_view
        self.guides_cam = guides_cam

    def get_warp(self, rays_o, rays_d, depthmap, tgt_trans: view.Trans, tgt_cam):
        rays_size = list(depthmap.size()) + [3]
        rays_o = rays_o.view(rays_size)
        rays_d = rays_d.view(rays_size)
        #print(rays_o.size(), rays_d.size(), depthmap.size())
        pcloud = rays_o + rays_d * depthmap[..., None]
        #print('pcloud', pcloud.size())
        pcloud_in_tgt = tgt_trans.trans_point(pcloud, inverse=True)
        # print(pcloud_in_tgt.size())
        pixel_positions = tgt_cam.proj(pcloud_in_tgt)
        pixel_positions[..., 0] /= tgt_cam.res[1] * 0.5
        pixel_positions[..., 1] /= tgt_cam.res[0] * 0.5
        pixel_positions -= 1
        return pixel_positions

    def refine_by_guide(self, image, depthmap, rays_o, rays_d, is_lr):
        if is_lr:
            image = nn_f.upsample(
                image[None, ...], scale_factor=2, mode='bicubic')[0]
            depthmap = nn_f.upsample(
                depthmap[None, None, ...], scale_factor=2, mode='bicubic')[0, 0]
        warp = self.get_warp(rays_o, rays_d, depthmap,
                             self.guides_view, self.guides_cam)
        warped_diff = nn_f.grid_sample(self.guides_diff, warp)
        print(warp.size(), warped_diff.size())
        avg_diff = torch.mean(warped_diff, 0)
        return image * (1 + avg_diff)


def constrast_enhance(image, sigma, fe):
    kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9
    mean = torch.cat([
        nn_f.conv2d(image[:, 0:1], kernel, padding=1),
        nn_f.conv2d(image[:, 1:2], kernel, padding=1),
        nn_f.conv2d(image[:, 2:3], kernel, padding=1)
    ], 1)
    cScale = 1.0 + sigma * fe
    return torch.clamp(mean + (image - mean) * cScale, 0, 1)


def morph_close(image: torch.Tensor):
    image_ = util.Tensor2MatImg(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_ = cv2.morphologyEx(image_, cv2.MORPH_CLOSE, kernel)
    return util.MatImg2Tensor(image_).to(image.device)


def get_grad(image: torch.Tensor, k=1, do_morph_close=False):
    kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          device=image.device, dtype=torch.float32).view(1, 1, 3, 3)
    x_grad = torch.cat([
        nn_f.conv2d(image[:, 0:1], kernel, padding=1),
        nn_f.conv2d(image[:, 1:2], kernel, padding=1),
        nn_f.conv2d(image[:, 2:3], kernel, padding=1)
    ], 1) / 4
    kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          device=image.device, dtype=torch.float32).view(1, 1, 3, 3)
    y_grad = torch.cat([
        nn_f.conv2d(image[:, 0:1], kernel, padding=1),
        nn_f.conv2d(image[:, 1:2], kernel, padding=1),
        nn_f.conv2d(image[:, 2:3], kernel, padding=1)
    ], 1) / 4
    grad = (x_grad ** 2 + y_grad ** 2).sqrt() * k
    if do_morph_close:
        grad = morph_close(grad)
    return grad.clamp(0, 1)


def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel.view(1, 1, 3, 3)


def grad_aware_median(image: torch.Tensor, median_kernel_size: int, grad_k: float,
                      grad_do_morph_close: bool):
    image_ = util.Tensor2MatImg(image)
    blur = cv2.medianBlur(image_, median_kernel_size)
    blur = util.MatImg2Tensor(blur).to(image.device)
    grad = get_grad(image, grad_k, grad_do_morph_close)
    return image * grad + blur * (1 - grad)


def grad_aware_gaussian(image, ksize, sigma=0):
    kernel = getGaussianKernel(ksize, sigma).to(image.device)
    print(kernel.size())
    blur = torch.cat([
        nn_f.conv2d(image[:, 0:1], kernel, padding=1),
        nn_f.conv2d(image[:, 1:2], kernel, padding=1),
        nn_f.conv2d(image[:, 2:3], kernel, padding=1)
    ], 1)
    grad = get_grad(image)
    return image * grad + blur * (1 - grad)


def bilateral_filter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = nn_f.pad(
        batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / \
        weights_color.sum(dim=(-1, -2), keepdim=True)

    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(
        *weights_space_dim).expand_as(weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix
