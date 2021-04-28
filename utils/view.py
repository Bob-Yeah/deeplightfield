
from typing import List, Mapping, Tuple, Union
import torch
import math
from . import misc


def fov2length(angle):
    return math.tan(math.radians(angle) / 2) * 2


class CameraParam(object):

    def __init__(self, params: Mapping[str, Union[float, bool]],
                 res: Tuple[int, int], *, device=None) -> None:
        super().__init__()
        params = self._convert_camera_params(params, res)
        self.res = res
        self.f = torch.tensor([params['fx'], params['fy'], 1], device=device)
        self.c = torch.tensor([params['cx'], params['cy']], device=device)
        self.device = device

    def to(self, device: torch.device):
        self.f = self.f.to(device)
        self.c = self.c.to(device)
        self.device = device
        return self

    def resize(self, res: Tuple[int, int]):
        self.f[0] = self.f[0] / self.res[1] * res[1]
        self.f[1] = self.f[1] / self.res[0] * res[0]
        self.c[0] = self.c[0] / self.res[1] * res[1]
        self.c[1] = self.c[1] / self.res[0] * res[0]
        self.res = res
        
    def proj(self, p: torch.Tensor, normalize=False, center_as_origin=False) -> torch.Tensor:
        """
        Project positions in local space to image plane

        :param p `Tensor(..., 3)`: positions in local space
        :param normalize: use normalized coord for image plane
        :param center_as_origin: take center as the origin if image plane instead of top-left corner
        :return `Tensor(..., 2)`: positions in image plane
        """
        p = p * self.f
        p = p[..., 0:2] / p[..., 2:3]
        if not center_as_origin:
            p = p + self.c
        if normalize:
            p = p / torch.tensor([self.res[1], self.res[0]], device=self.device)
        return p

    def unproj(self, p: torch.Tensor, z: torch.Tensor = None, normalize=False, center_as_origin=False) -> torch.Tensor:
        """
        Unproject positions in image plane to local space

        :param p `Tensor(..., 2)`: positions in image plane
        :param z `Tensor(..., 1)`: depths of positions, None means all depths set to 1
        :param normalize: use normalized coord for image plane
        :param center_as_origin: take center as the origin if image plane instead of top-left corner
        :return: positions in local space
        """
        if normalize:
            p = p * torch.tensor([self.res[1], self.res[0]], device=self.device)
        if not center_as_origin:
            p = p - self.c
        p = misc.broadcast_cat(p / self.f[0:2], 1.0)
        if z != None:
            p = p * z
        return p

    def get_local_rays(self, flatten=False, norm=True) -> torch.Tensor:
        """
        Get view rays in local space

        :param flatten: whether flatten the return tensor
        :param norm: whether normalize rays to unit length
        :return `Tensor(H, W, 3)|Tensor(HW, 3)`: the shape is determined by parameter 'flatten'
        """
        coords = misc.meshgrid(*self.res).to(self.f.device)
        rays = self.unproj(coords)
        if norm:
            rays = rays / rays.norm(dim=-1, keepdim=True)
        if flatten:
            rays = rays.flatten(0, 1)
        return rays

    def get_global_rays(self, trans, flatten=False, norm=True) -> torch.Tensor:
        """
        [summary]

        :param t `Tensor(N.., 3)`: translation vectors
        :param r `Tensor(N.., 3, 3)`: rotation matrices
        :param flatten: [description], defaults to False
        :param norm: [description], defaults to True
        :return: [description]
        """
        rays = self.get_local_rays(flatten, norm)  # (M.., 3)
        rays_o, _ = torch.broadcast_tensors(trans.t[..., None, :], rays) if flatten \
            else torch.broadcast_tensors(trans.t[..., None, None, :], rays)  # (N.., M.., 3)
        rays_d = trans.trans_vector(rays)
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
            if input_is_normalized:
                camera_params['fy'] = 1 / fov2length(input_camera_params['fov'])
                camera_params['fx'] = camera_params['fy'] / view_res[1] * view_res[0]
            else:
                camera_params['fx'] = camera_params['fy'] = view_res[0] / \
                    fov2length(input_camera_params['fov'])
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


class Trans(object):

    def __init__(self, t: torch.Tensor, r: torch.Tensor) -> None:
        self.t = t
        self.r = r
        if len(self.t.size()) == 1:
            self.t = self.t[None, :]
            self.r = self.r[None, :, :]

    def trans_point(self, p: torch.Tensor, inverse=False) -> torch.Tensor:
        """
        Transform points by given translation vectors and rotation matrices

        :param p `Tensor(N.., 3)`: points to transform
        :param t `Tensor(M.., 3)`: translation vectors
        :param r `Tensor(M.., 3, 3)`: rotation matrices
        :param inverse: whether perform inverse transform
        :return `Tensor(M.., N.., 3)`: transformed points
        """
        size_N = list(p.size())[:-1]
        size_M = list(self.r.size())[:-2]
        out_size = size_M + size_N + [3]
        t_size = size_M + [1 for _ in range(len(size_N))] + [3]
        t = self.t.view(t_size) # (M.., 1.., 3)
        if inverse:
            p = (p - t).view(size_M + [-1, 3])
            r = self.r
        else:
            p = p.view(-1, 3)
            r = self.r.movedim(-1, -2) # Transpose rotation matrices
        out = torch.matmul(p, r).view(out_size)
        if not inverse:
            out = out + t
        return out

    def trans_vector(self, v: torch.Tensor, inverse=False) -> torch.Tensor:
        """
        Transform vectors by given translation vectors and rotation matrices

        :param v `Tensor(N.., 3)`: vectors to transform
        :param r `Tensor(M.., 3, 3)`: rotation matrices
        :param inverse: whether perform inverse transform
        :return `Tensor(M.., N.., 3)`: transformed vectors
        """
        out_size = list(self.r.size())[:-2] + list(v.size())[:-1] + [3]
        r = self.r if inverse else self.r.movedim(-1, -2) # Transpose rotation matrices
        out = torch.matmul(v.view(-1, 3), r).view(out_size)
        return out
    
    def size(self) -> List[int]:
        return list(self.t.size()[:-1])
    
    def get(self, *index):
        return Trans(self.t[index], self.r[index])


def trans_point(p: torch.Tensor, t: torch.Tensor, r: torch.Tensor, inverse=False) -> torch.Tensor:
    """
    Transform points by given translation vectors and rotation matrices

    :param p `Tensor(N.., 3)`: points to transform
    :param t `Tensor(M.., 3)`: translation vectors
    :param r `Tensor(M.., 3, 3)`: rotation matrices
    :param inverse: whether perform inverse transform
    :return `Tensor(M.., N.., 3)`: transformed points
    """
    size_N = list(p.size())[0:-1]
    size_M = list(r.size())[0:-2]
    out_size = size_M + size_N + [3]
    t_size = size_M + [1 for _ in range(len(size_N))] + [3]
    t = t.view(t_size)
    if not inverse:
        r = r.movedim(-1, -2)  # Transpose rotation matrices
    else:
        p = p - t
    out = torch.matmul(p.view(size_M + [-1, 3]), r)
    out = out.view(out_size)
    if not inverse:
        out = out + t
    return out


def trans_vector(v: torch.Tensor, r: torch.Tensor, inverse=False) -> torch.Tensor:
    """
    Transform vectors by given translation vectors and rotation matrices

    :param v `Tensor(N.., 3)`: vectors to transform
    :param r `Tensor(M.., 3, 3)`: rotation matrices
    :param inverse: whether perform inverse transform
    :return `Tensor(M.., N.., 3)`: transformed vectors
    """
    out_size = list(r.size())[0:-2] + list(v.size())[0:-1] + [3]
    if not inverse:
        r = r.movedim(-1, -2)  # Transpose rotation matrices
    out = torch.matmul(v.flatten(0, -2), r).view(out_size)
    return out

