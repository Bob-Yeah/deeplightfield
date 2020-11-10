import matplotlib.pyplot as plt
import numpy as np
import torch

def RandomGenSamplesInPupil(conf, n_samples):
    '''
    Random sample n_samples positions in pupil region
    
    Parameters
    --------
    conf      - multi-layers' parameters configuration
    n_samples - number of samples to generate
    
    Returns
    --------
    a n_samples x 2 tensor with 2D sample position in each row
    '''
    samples = torch.empty(n_samples, 2)
    i = 0
    while i < n_samples:
        s = (torch.rand(2) - 0.5) * conf.pupil_size
        if np.linalg.norm(s) > conf.pupil_size / 2.:
            continue
        samples[i, :] = s
        i += 1
    return samples

def GenSamplesInPupil(conf, circles):
    '''
    Sample positions on circles in pupil region
    
    Parameters
    --------
    conf      - multi-layers' parameters configuration
    circles   - number of circles to sample
    
    Returns
    --------
    a n_samples x 2 tensor with 2D sample position in each row
    '''
    samples = torch.tensor([[ 0., 0. ]])
    for i in range(1, circles):
        r = conf.pupil_size / 2. / (circles - 1) * i
        n = 4 * i
        for j in range(0, n):
            angle = 2 * np.pi / n * j
            samples = torch.cat((samples, torch.tensor([[ r * np.cos(angle), r * np.sin(angle)]])),dim=0)
    return samples

def GenRetinal2LayerMappings(conf, df, v, u):
    '''
    Generate the mapping matrix from retinal to layers.
    
    Parameters
    --------
    conf - multi-layers' parameters configuration
    df   - focal distance
    v    - a 1 x 2 tensor stores half viewport
    u    - a M x 2 tensor stores M sample positions on pupil
    
    Returns
    --------
    The mapping matrix
    '''
    H_r = conf.retinal_res[0]
    W_r = conf.retinal_res[1]
    D_r = conf.retinal_res.double()
    N = conf.n_layers
    M = u.size()[0] #41
    Phi = torch.empty(H_r, W_r, N, M, 2, dtype=torch.long)
    p_rx, p_ry = torch.meshgrid(torch.tensor(range(0, H_r)),
                                torch.tensor(range(0, W_r)))
    p_r = torch.stack([p_rx, p_ry], 2).unsqueeze(2).expand(-1, -1, M, -1)
    # print(p_r.shape) #torch.Size([480, 640, 41, 2])
    for i in range(0, N):
        dpi = conf.h_layer[i] / conf.layer_res[0] # 1 / 480
        ci = conf.layer_res / 2 # [240,320]
        di = conf.d_layer[i] # 深度
        pi_r = di * v * (1. / D_r * (p_r + 0.5) - 0.5) / dpi # [480, 640, 41, 2]
        wi = (1 - di / df) / dpi # (1 - 深度/聚焦) / dpi  df = 2.625 di = 1.75
        pi = torch.floor(pi_r + ci + wi * u)
        torch.clamp_(pi[:, :, :, 0], 0, conf.layer_res[0] - 1)
        torch.clamp_(pi[:, :, :, 1], 0, conf.layer_res[1] - 1)
        Phi[:, :, i, :, :] = pi
    return Phi

def GenRetinalFromLayers(layers, Phi):
    # layers:  2, color, height, width 
    # Phi:torch.Size([480, 640, 2, 41, 2])
    M = Phi.size()[3] # 41
    N = Phi.size()[2] # 2
    # print(layers.shape)# torch.Size([2, 3, 480, 640])
    # print(Phi.shape)# torch.Size([480, 640, 2, 41, 2])
    # retinal image: 3channels x retinal_size
    retinal = torch.zeros(3, Phi.size()[0], Phi.size()[1])
    for j in range(0, M):
        retinal_view = torch.zeros(3, Phi.size()[0], Phi.size()[1])
        for i in range(0, N):
            retinal_view.add_(layers[i,:, Phi[:, :, i, j, 0], Phi[:, :, i, j, 1]])
        retinal.add_(retinal_view)
    retinal.div_(M)
    return retinal


    