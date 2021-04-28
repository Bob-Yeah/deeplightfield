import torch
import torch.nn.functional as nn_f
from typing import Any, List, Mapping, Tuple
from torch import nn
from utils import view
from utils import misc
from utils.perf import Perf
from . import refine
from .foveation import Foveation


class GenFinal(object):

    def __init__(self, layers_fov: List[float],
                 layers_res: List[Tuple[int, int]],
                 full_res: Tuple[int, int],
                 fovea_net: nn.Module,
                 periph_net: nn.Module,
                 device: torch.device = None) -> None:
        super().__init__()
        self.layer_cams = [
            view.CameraParam({
                'fov': layers_fov[i],
                'cx': 0.5,
                'cy': 0.5,
                'normalized': True
            }, layers_res[i], device=device)
            for i in range(len(layers_fov))
        ]
        self.full_cam = view.CameraParam({
            'fov': layers_fov[-1],
            'cx': 0.5,
            'cy': 0.5,
            'normalized': True
        }, full_res, device=device)
        self.fovea_net = fovea_net.to(device)
        self.periph_net = periph_net.to(device)
        self.foveation = Foveation(layers_fov, full_res, device=device)
        self.device = device

    def to(self, device: torch.device):
        self.fovea_net.to(device)
        self.periph_net.to(device)
        self.foveation.to(device)
        self.full_cam.to(device)
        for cam in self.layer_cams:
            cam.to(device)
        self.device = device
        return self

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.gen(*args, **kwds)

    def gen(self, gaze, trans: view.Trans, *,
            mono_trans: view.Trans = None,
            shift: int = 0,
            warp_by_depth: bool = False,
            ret_raw=False,
            perf_time=False) -> Mapping[str, torch.Tensor]:
        fovea_cam = self._adjust_cam(self.layer_cams[0], self.full_cam, gaze)
        mid_cam = self._adjust_cam(self.layer_cams[1], self.full_cam, gaze)
        periph_cam = self.layer_cams[2]
        trans_periph = mono_trans if mono_trans != None else trans

        perf = Perf(True, True) if perf_time else None

        # *_rays_o, *_rays_d: (1, N, 3)
        fovea_rays_o, fovea_rays_d = fovea_cam.get_global_rays(trans, True)
        mid_rays_o, mid_rays_d = mid_cam.get_global_rays(trans_periph, True)
        periph_rays_o, periph_rays_d = periph_cam.get_global_rays(
            trans_periph, True)
        mid_periph_rays_o = torch.cat([mid_rays_o, periph_rays_o], 1)
        mid_periph_rays_d = torch.cat([mid_rays_d, periph_rays_d], 1)
        if perf_time:
            perf.checkpoint('Get rays')

        perf1 = Perf(True, True) if perf_time else None

        fovea_inferred, fovea_depth = self._infer(
            self.fovea_net, fovea_rays_o, fovea_rays_d, [fovea_cam.res], True)

        if perf_time:
            perf1.checkpoint('Infer fovea')
        mid_inferred, mid_depth, periph_inferred, periph_depth = self._infer(
            self.periph_net, mid_periph_rays_o, mid_periph_rays_d,
            [mid_cam.res, periph_cam.res], True)

        if perf_time:
            perf1.checkpoint('Infer mid & periph')
            perf.checkpoint('Infer')

        if mono_trans != None and shift == 0:  # do warp
            fovea_depth[torch.isnan(fovea_depth)] = 50
            mid_depth[torch.isnan(mid_depth)] = 50
            periph_depth[torch.isnan(periph_depth)] = 50

            if warp_by_depth:
                z_list = misc.depth_sample((1, 50), 4, True)
                mid_inferred = self._warp(trans, mono_trans, mid_cam,
                                          z_list, mid_inferred, mid_depth)
                periph_inferred = self._warp(trans, mono_trans, periph_cam,
                                             z_list, periph_inferred, periph_depth)
                if perf_time:
                    perf.checkpoint('Mono warp')
            else:   
                p = torch.tensor([[0, 0, torch.mean(fovea_depth)]],
                                device=self.device)
                p_ = trans.trans_point(mono_trans.trans_point(p), inverse=True)
                shift = self.full_cam.proj(
                    p_, center_as_origin=True)[..., 0].item()
                shift = round(shift)
                if perf_time:
                    perf.checkpoint('Mono shift')

        fovea_refined = refine.grad_aware_median(fovea_inferred, 3, 3, True)
        fovea_refined = refine.constrast_enhance(fovea_refined, 3, 0.2)
        mid_refined = refine.constrast_enhance(mid_inferred, 5, 0.2)
        periph_refined = refine.constrast_enhance(periph_inferred, 5, 0.2)

        if perf_time:
            perf.checkpoint('Refine')

        blended = self.foveation.synthesis([
            fovea_refined,
            mid_refined,
            periph_refined
        ], (gaze[0], gaze[1]), [0, shift, shift] if shift != 0 else None)

        if perf_time:
            perf.checkpoint('Blend')

        if ret_raw:
            return {
                'fovea': fovea_refined,
                'mid': mid_refined,
                'periph': periph_refined,
                'blended': blended,
                'fovea_raw': fovea_inferred,
                'mid_raw': mid_inferred,
                'periph_raw': periph_inferred,
                'blended_raw': self.foveation.synthesis([
                    fovea_inferred,
                    mid_inferred,
                    periph_inferred
                ], (gaze[0], gaze[1]))
            }
        return {
            'fovea': fovea_refined,
            'mid': mid_refined,
            'periph': periph_refined,
            'blended': blended
        }

    def _infer(self, net, rays_o: torch.Tensor, rays_d: torch.Tensor,
               res_list: List[Tuple[int, int]], ret_depth=False):
        if ret_depth:
            colors, depths = net(rays_o.view(-1, 3), rays_d.view(-1, 3),
                                 ret_depth=True)
        else:
            colors = net(rays_o.view(-1, 3), rays_d.view(-1, 3))
            depths = None
        images = []
        offset = 0
        for res in res_list:
            bound = offset + res[0] * res[1]
            images.append(colors[offset:bound].view(
                1, res[0], res[1], -1).permute(0, 3, 1, 2))
            if ret_depth:
                images.append(depths[offset:bound].view(
                    1, res[0], res[1]))
            offset = bound
        return tuple(images)

    def _adjust_cam(self, cam: view.CameraParam, full_cam: view.CameraParam,
                    gaze: Tuple[float, float]) -> view.CameraParam:
        fovea_offset = (
            (gaze[0]) / full_cam.f[0].item() * cam.f[0].item(),
            (gaze[1]) / full_cam.f[1].item() * cam.f[1].item()
        )
        return view.CameraParam({
            'fx': cam.f[0].item(),
            'fy': cam.f[1].item(),
            'cx': cam.c[0].item() - fovea_offset[0],
            'cy': cam.c[1].item() - fovea_offset[1]
        }, cam.res, device=self.device)

    def _warp(self, trans: view.Trans, trans0: view.Trans,
              cam: view.CameraParam, z_list: torch.Tensor,
              image: torch.Tensor, depthmap: torch.Tensor) -> torch.Tensor:
        """
        [summary]

        :param trans: [description]
        :param trans0: [description]
        :param cam: [description]
        :param z_list: [description]
        :param image `Tensor(B, C, H, W)`:
        :param depthmap `Tensor(B, H, W)`:
        :return `Tensor(B, C, H, W)`:
        """
        B = image.size(0)
        rays_d = cam.get_global_rays(trans, norm=False)[1]  # (1, H, W, 3)
        rays_d_0 = trans0.trans_vector(rays_d, True)[0]  # (1, H, W, 3)
        t_0 = trans0.trans_point(trans.t, True)[0]  # (1, 3)
        q1_0 = torch.empty(B, cam.res[0], cam.res[1],
                           3, device=cam.device)  # near
        q2_0 = torch.empty(B, cam.res[0], cam.res[1],
                           3, device=cam.device)  # far
        determined = torch.zeros(B, cam.res[0], cam.res[1], 1,
                                 dtype=torch.bool, device=cam.device)
        for z in z_list:
            p_0 = rays_d_0 * z + t_0  # (1, H, W, 3)
            d_of_p_0 = torch.norm(p_0 - trans0.t, dim=-1,
                                  keepdim=True)  # (1, H, W, 1)
            v_of_p_0 = p_0 / d_of_p_0  # (1, H, W, 3)
            coords = cam.proj(p_0, True) * 2 - 1  # (1, H, W, 2)
            d = nn_f.grid_sample(
                depthmap[:, None, :, :],
                coords.expand(B, -1, -1, -1)).permute(0, 2, 3, 1)  # (B, H, W, 1)
            q = v_of_p_0 * d  # (B, H, W, 3)
            near_selector = d < d_of_p_0
            # Fill q2(far) when undetermined and d > d_of_p_0
            q2_selector = (~determined & ~near_selector).expand(-1, -1, -1, 3)
            q2_0[q2_selector] = q[q2_selector]
            # Fill q1(near) when undetermined and d <= d_of_p_0
            q1_selector = (~determined & near_selector).expand(-1, -1, -1, 3)
            q1_0[q1_selector] = q[q1_selector]
            # Mark as determined for d0 <= d
            determined[near_selector] = True

        # Compute intersection x of q1-q2 and rays (in trans0 space)
        k = torch.cross(q1_0 - t_0, rays_d_0, dim=-1).norm(dim=-1, keepdim=True) / \
            torch.cross(rays_d_0, q2_0 - t_0, dim=-1).norm(dim=-
                                                           1, keepdim=True)  # (B, H, W, 1)
        x_0 = (q2_0 - q1_0) * k / (k + 1) + q1_0
        coords = cam.proj(x_0, True) * 2 - 1  # (B, H, W, 2)
        return nn_f.grid_sample(image, coords)
