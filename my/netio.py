from typing import Mapping
import torch
import numpy as np

def PrintNet(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)


def LoadNet(path, model, solver=None, discriminator=None):
    print('Load net from %s ...' % path)
    whole_dict: Mapping = torch.load(path)
    model.load_state_dict(whole_dict['model'])
    if solver:
        solver.load_state_dict(whole_dict['solver'])
    if discriminator:
        discriminator.load_state_dict(whole_dict['discriminator'])
    return whole_dict['iters'] if 'iters' in whole_dict else 0
    

def SaveNet(path, model, solver=None, discriminator=None, iters=None):
    print('Saving net to %s ...' % path)
    whole_dict = {
        'model': model.state_dict()
    }
    if solver:
        whole_dict.update({'solver': solver.state_dict()})
    if discriminator:
        whole_dict.update({'discriminator': discriminator.state_dict()})
    if iters:
        whole_dict.update({'iters': iters})
    torch.save(whole_dict, path)