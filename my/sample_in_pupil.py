import torch
import numpy as np


def RandomGen(pupil_size: float, n_samples: int) -> torch.Tensor:
    """
    Random sample n_samples positions in pupil region
    
    :param pupil_size: multi-layers' parameters configuration
    :param n_samples:  number of samples to generate

    :return: n_samples x 3, with 3D sample position in each row
    """
    samples = torch.empty(n_samples, 3)
    i = 0
    while i < n_samples:
        s = (torch.rand(2) - 0.5) * pupil_size
        if np.linalg.norm(s) > pupil_size / 2.:
            continue
        samples[i, :] = [s[0], s[1], 0]
        i += 1
    return samples


def CircleGen(pupil_size: float, circles: int) -> torch.Tensor:
    """
    Sample positions on circles in pupil region

    :param pupil_size: diameter of pupil
    :param circles:    number of circles to sample
    
    :return: M x 3, with 3D sample position in each row
    """
    samples = torch.zeros(1, 3)
    for i in range(1, circles):
        r = pupil_size / 2. / (circles - 1) * i
        n = 4 * i
        for j in range(0, n):
            angle = 2 * np.pi / n * j
            samples = torch.cat([samples, torch.tensor([[r * np.cos(angle), r * np.sin(angle), 0]])], 0)
    return samples
