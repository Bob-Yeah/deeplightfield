import torch
from torch import nn
from typing import List, Mapping, Tuple
from . import view
from . import refine
from .foveation import Foveation
from .simple_perf import SimplePerf


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
        self.foveation = Foveation(
            layers_fov, full_res, device=device)
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

    def gen(self, gaze, trans, ret_raw=False, perf_time=False) -> Mapping[str, torch.Tensor]:
        fovea_cam = self._adjust_cam(self.layer_cams[0], self.full_cam, gaze)
        mid_cam = self._adjust_cam(self.layer_cams[1], self.full_cam, gaze)
        periph_cam = self.layer_cams[2]

        perf = SimplePerf(True, True) if perf_time else None

        # x_rays_o, x_rays_d: (Hx, Wx, 3)
        fovea_rays_o, fovea_rays_d = fovea_cam.get_global_rays(trans, True)
        mid_rays_o, mid_rays_d = mid_cam.get_global_rays(trans, True)
        periph_rays_o, periph_rays_d = periph_cam.get_global_rays(trans, True)
        mid_periph_rays_o = torch.cat([mid_rays_o, periph_rays_o], 1)
        mid_periph_rays_d = torch.cat([mid_rays_d, periph_rays_d], 1)
        if perf_time:
            perf.Checkpoint('Get rays')

        perf1 = SimplePerf(True, True) if perf_time else None

        fovea_inferred = self.fovea_net(fovea_rays_o[0], fovea_rays_d[0]).view(
            1, fovea_cam.res[0], fovea_cam.res[1], -1).permute(0, 3, 1, 2)  # (1, C, H_fovea, W_fovea)
        if perf_time:
            perf1.Checkpoint('Infer fovea')

        periph_mid_inferred = self.periph_net(mid_periph_rays_o[0], mid_periph_rays_d[0])
        mid_inferred = periph_mid_inferred[:mid_cam.res[0] * mid_cam.res[1], :].view(
            1, mid_cam.res[0], mid_cam.res[1], -1).permute(0, 3, 1, 2)
        periph_inferred = periph_mid_inferred[mid_cam.res[0] * mid_cam.res[1]:, :].view(
            1, periph_cam.res[0], periph_cam.res[1], -1).permute(0, 3, 1, 2)
        if perf_time:
            perf1.Checkpoint('Infer mid & periph')
            perf.Checkpoint('Infer')

        fovea_refined = refine.constrast_enhance(fovea_inferred, 3, 0.2)
        mid_refined = refine.constrast_enhance(mid_inferred, 5, 0.2)
        periph_refined = refine.constrast_enhance(periph_inferred, 5, 0.2)

        if perf_time:
            perf.Checkpoint('Refine')

        blended = self.foveation.synthesis([
            fovea_refined,
            mid_refined,
            periph_refined
        ], (gaze[0], gaze[1]))

        if perf_time:
            perf.Checkpoint('Blend')

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
