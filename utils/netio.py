import torch
import numpy as np
from utils import device


def log(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)


def load(path, model, **extra_models):
    print('Load net from %s ...' % path)
    whole_dict = torch.load(path, map_location=device.default())
    model.load_state_dict(whole_dict['model'])
    for model, key in extra_models:
        if key in whole_dict:
            model.load_state_dict(whole_dict[key])
    return whole_dict['iters'] if 'iters' in whole_dict else 0


def save(path, model, iters, print_log=True, **extra_models):
    if print_log:
        print('Saving net to %s ...' % path)
    whole_dict = {
        'iters': iters,
        'model': model.state_dict(),
    }
    for model, key in extra_models:
        whole_dict[key] = model.state_dict()
    torch.save(whole_dict, path)
