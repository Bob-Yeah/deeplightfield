from typing import Mapping, Tuple, Union
import torch
from . import util


class CameraParam(object):

    def __init__(self, params: Mapping[str, Union[float, bool]],
                 res: Tuple[int, int], *, device=None) -> None:
        super().__init__()
        params = self._convert_camera_params(params, res)
        self.res = res
        self.f = torch.tensor([params['fx'], params['fy'], 1], device=device)
        self.c = torch.tensor([params['cx'], params['cy']], device=device)

    def to(self, device: torch.device):
        self.f = self.f.to(device)
        self.c = self.c.to(device)
        return self

    def proj(self, p: torch.Tensor) -> torch.Tensor:
        """
        Project positions in local space to image plane

        :param p ```Tensor(..., 3)```: positions in local space
        :return ```Tensor(..., 2)```: positions in image plane
        """
        p = p * self.f
        p = p[..., 0:2] / p[..., 2:3] + self.c
        return p

    def unproj(self, p: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Unproject positions in image plane to local space

        :param p ```Tensor(..., 2)```: positions in image plane
        :param z ```Tensor(..., 1)```: depths of positions, None means all depths set to 1
        :return: positions in local space
        """
        p = util.broadcast_cat((p - self.c) / self.f[0:2], 1.0)
        if z != None:
            p = p * z
        return p

    def get_local_rays(self, flatten=False, norm=True) -> torch.Tensor:
        """
        Get view rays in local space

        :param flatten: whether flatten the return tensor
        :param norm: whether normalize rays to unit length
        :return ```Tensor(H, W, 3)|Tensor(HW, 3)```: the shape is determined by parameter 'flatten'
        """
        coords = util.MeshGrid(self.res).to(self.f.device)
        rays = self.unproj(coords)
        if norm:
            rays = rays / rays.norm(dim=-1, keepdim=True)
        if flatten:
            rays = rays.flatten(0, 1)
        return rays

    def get_global_rays(self, t: torch.Tensor, r: torch.Tensor,
                        flatten=False, norm=True) -> torch.Tensor:
        """
        [summary]

        :param t ```Tensor(N.., 3)```: translation vectors
        :param r ```Tensor(N.., 3, 3)```: rotation matrices
        :param flatten: [description], defaults to False
        :param norm: [description], defaults to True
        :return: [description]
        """
        rays = self.get_local_rays(flatten, norm)  # (M.., 3)
        rays_o, _ = torch.broadcast_tensors(
            t[..., None, None, :], rays)  # (N.., M.., 3)
        rays_d = trans_vector(rays, r)
        return rays_o, rays_d

    def _convert_camera_params(self, input_camera_params: Mapping[str, Union[float, bool]],
                               view_res: Tuple[int, int]) -> Mapping[str, Union[float, bool]]:
        """
        Check and convert camera parameters in config file to pixel-space

        :param cam_params: { ["fx", "fy" | "fov"], "cx", "cy", ["normalized"] },
            the parameters of camera
        :return: camera parameters
        """
        input_is_normalized = bool(input_camera_params.get('normalized'))
        camera_params = {}
        if 'fov' in input_camera_params:
            camera_params['fx'] = camera_params['fy'] = \
                (1 if input_is_normalized else view_res[0]) / \
                util.Fov2Length(input_camera_params['fov'])
            camera_params['fy'] *= -1
        else:
            camera_params['fx'] = input_camera_params['fx']
            camera_params['fy'] = input_camera_params['fy']
        camera_params['cx'] = input_camera_params['cx']
        camera_params['cy'] = input_camera_params['cy']
        if input_is_normalized:
            camera_params['fx'] *= view_res[1]
            camera_params['fy'] *= view_res[0]
            camera_params['cx'] *= view_res[1]
            camera_params['cy'] *= view_res[0]
        return camera_params


def trans_point(p: torch.Tensor, t: torch.Tensor, r: torch.Tensor, inverse=False) -> torch.Tensor:
    """
    Transform points by given translation vectors and rotation matrices

    :param p ```Tensor(N.., 3)```: points to transform
    :param t ```Tensor(M.., 3)```: translation vectors
    :param r ```Tensor(M.., 3, 3)```: rotation matrices
    :param inverse: whether perform inverse transform
    :return ```Tensor(M.., N.., 3)```: transformed points
    """
    out_size = list(r.size())[0:-2] + list(p.size())[0:-1] + [3]
    t_size = list(t.size()[0:-1]) + \
        [1 for _ in range(len(p.size()[0:-1]))] + [3]
    t = t.view(t_size)
    if not inverse:
        r = r.movedim(-1, -2)  # Transpose rotation matrices
    else:
        p = p - t
    out = torch.matmul(p.flatten(0, -2), r).view(out_size)
    if not inverse:
        out = out + t
    return out


def trans_vector(v: torch.Tensor, r: torch.Tensor, inverse=False) -> torch.Tensor:
    """
    Transform vectors by given translation vectors and rotation matrices

    :param v ```Tensor(N.., 3)```: vectors to transform
    :param r ```Tensor(M.., 3, 3)```: rotation matrices
    :param inverse: whether perform inverse transform
    :return ```Tensor(M.., N.., 3)```: transformed vectors
    """
    out_size = list(r.size())[0:-2] + list(v.size())[0:-1] + [3]
    if not inverse:
        r = r.movedim(-1, -2)  # Transpose rotation matrices
    out = torch.matmul(v.flatten(0, -2), r).view(out_size)
    return out
