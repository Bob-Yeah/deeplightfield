import numpy as np
from .constants import *


def helix(t_range, loops, n):
    n_per_loop = n // loops
    angles = np.linspace(0, 2 * PI, n_per_loop, endpoint=False)[None, :]. \
        repeat(loops, axis=0).flatten()
    centers = np.empty([n, 3])
    centers[:, 0] = 0.5 * t_range[0] * np.cos(angles)
    centers[:, 1] = 0.5 * t_range[1] * np.sin(angles)
    centers[:, 2] = 0.5 * t_range[2] * np.concatenate([
        np.linspace(-1, 1, n // 2),
        np.linspace(1, -1, n - n // 2)
    ])
    rots = np.zeros([n, 2])
    return centers, rots


def scan_around(t_range, circles, n):
    angles = np.linspace(-PI, PI, n // circles, endpoint=False)
    x_rots = angles[None, :].repeat(circles, axis=0).flatten()
    c_angles = angles + 0.8 * PI
    centers = np.empty([n, 3])
    for i in range(circles):
        r = (0.5 * t_range[0] / circles * (i + 1),
             0.5 * t_range[1] / circles * (i + 1),
             0.5 * t_range[2])
        s = slice(i * len(angles), (i + 1) * len(angles))
        centers[s, 0] = r[0] * np.sin(c_angles)
        centers[s, 1] = r[1] * np.sin(angles * 10 + i * 2 * PI / circles)
        centers[s, 2] = r[2] * np.cos(c_angles)
    rots = np.stack([x_rots, np.zeros_like(x_rots)], axis=1)
    return centers, rots


def look_around(t_range, n):
    angles = np.linspace(-PI, PI, n, endpoint=False)
    x_rots = angles
    c_angles = angles + 0.8 * PI
    centers = np.empty([n, 3])
    r = (0.5 * t_range[0], 0.5 * t_range[1], 0.5 * t_range[2])
    centers[:, 0] = r[0] * np.sin(c_angles)
    centers[:, 1] = r[1] * np.sin(angles * 4)
    centers[:, 2] = r[2] * np.cos(c_angles)
    rots = np.stack([x_rots, np.zeros_like(x_rots)], axis=1)
    return centers, rots