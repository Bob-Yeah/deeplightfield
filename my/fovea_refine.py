import torch
import torch.nn.functional as nn_f
from . import view


def get_warp(rays_o, rays_d, depthmap, tgt_o, tgt_r, tgt_cam):
    print(rays_o.size(), rays_d.size(), depthmap.size())
    pcloud = rays_o + rays_d * depthmap[..., None]
    print(rays_o.size(), rays_d.size(), depthmap.size(), pcloud.size())
    pcloud_in_tgt = view.trans_point(
        pcloud, tgt_o, tgt_r, inverse=True)
    print(pcloud_in_tgt.size())
    pixel_positions = tgt_cam.proj(pcloud_in_tgt)
    pixel_positions[..., 0] /= tgt_cam.res[1] * 0.5
    pixel_positions[..., 1] /= tgt_cam.res[0] * 0.5
    pixel_positions -= 1
    return pixel_positions


def refine(image, depthmap, rays_o, rays_d, bounds_img, bounds_o,
           bounds_r, ref_cam: view.CameraParam, net, is_lr):
    if is_lr:
        image = nn_f.upsample(
            image[None, ...], scale_factor=2, mode='bicubic')[0]
        depthmap = nn_f.upsample(
            depthmap[None, None, ...], scale_factor=2, mode='bicubic')[0, 0]
    bounds_rays_o, bounds_rays_d = ref_cam.get_global_rays(
        bounds_o, bounds_r, flatten=True)
    bounds_inferred = torch.stack([
        net(bounds_rays_o[i], bounds_rays_d[i]).view(
            ref_cam.res[0], ref_cam.res[1], -1).permute(2, 0, 1)
        for i in range(bounds_img.size(0))
    ], 0)
    bounds_diff = (bounds_img - bounds_inferred) / (bounds_inferred + 1e-5)
    bounds_warp = get_warp(rays_o, rays_d, depthmap,
                           bounds_o, bounds_r, ref_cam)
    warped_diff = nn_f.grid_sample(bounds_diff, bounds_warp)
    print(bounds_warp.size(), warped_diff.size())
    avg_diff = torch.mean(warped_diff, 0)
    return image * (1 + avg_diff)
