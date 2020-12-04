import matplotlib.pyplot as plt
import numpy as np
import torch
import glm

def Fov2Length(angle):
    '''

    '''
    return np.tan(angle * np.pi / 360) * 2


def RandomGenSamplesInPupil(pupil_size, n_samples):
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
        s = (torch.rand(2) - 0.5) * pupil_size
        if np.linalg.norm(s) > pupil_size / 2.:
            continue
        samples[i, :] = [ s[0], s[1], 0 ]
        i += 1
    return samples

def GenSamplesInPupil(pupil_size, circles):
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
    samples = torch.zeros(1, 3)
    for i in range(1, circles):
        r = pupil_size / 2. / (circles - 1) * i
        n = 4 * i
        for j in range(0, n):
            angle = 2 * np.pi / n * j
            samples = torch.cat([ samples, torch.tensor([[ r * np.cos(angle), r * np.sin(angle), 0 ]]) ], 0)
    return samples

class RetinalGen(object):
    '''
    Class for retinal generation process
    
    Properties
    --------
    conf - multi-layers' parameters configuration
    u    - M x 3 tensor, M sample positions in pupil
    p_r  - H_r x W_r x 3 tensor, retinal pixel grid, [H_r, W_r] is the retinal resolution
    Phi  - N x H_r x W_r x M x 2 tensor, retinal to layers mapping, N is number of layers
    mask - N x H_r x W_r x M x 2 tensor, indicates invalid (out-of-range) mapping
    
    Methods
    --------
    '''
    def __init__(self, conf, u):
        '''
        Initialize retinal generator instance

        Parameters
        --------
        conf - multi-layers' parameters configuration
        u    - a M x 3 tensor stores M sample positions in pupil
        '''
        self.conf = conf
        # self.u = u.to(cuda_dev)
        self.u = u # M x 3 M sample positions 
        self.D_r = conf.retinal_res # retinal res 480 x 640 
        self.N = conf.GetNLayers() # 2 
        self.M = u.size()[0] # samples
        p_rx, p_ry = torch.meshgrid(torch.tensor(range(0, self.D_r[0])),
                                    torch.tensor(range(0, self.D_r[1])))
        self.p_r = torch.cat([
            ((torch.stack([p_rx, p_ry], 2) + 0.5) / self.D_r - 0.5) * conf.GetEyeViewportSize(), # 眼球视野
            torch.ones(self.D_r[0], self.D_r[1], 1)
        ], 2)

        # self.Phi = torch.empty(N, D_r[0], D_r[1], M, 2, device=cuda_dev, dtype=torch.long)
        # self.mask = torch.empty(self.N, self.D_r[0], self.D_r[1], self.M, 2, dtype=torch.float) # 2 x 480 x 640 x 41 x 2
        
    def CalculateRetinal2LayerMappings(self, df, gaze):
        '''
        Calculate the mapping matrix from retinal to layers.

        Parameters
        --------
        df   - focus distance
        gaze - 2 x 1 tensor, eye rotation angle (degs) in horizontal and vertical direction

        '''
        Phi = torch.empty(self.N, self.D_r[0], self.D_r[1], self.M, 2, dtype=torch.long) # 2 x 480 x 640 x 41 x 2
        mask = torch.empty(self.N, self.D_r[0], self.D_r[1], self.M, 2, dtype=torch.float)
        D_r = self.conf.retinal_res        # D_r: Resolution of retinal 480 640
        V = self.conf.GetEyeViewportSize() # V: Viewport size of eye 
        c = (self.conf.layer_res / 2)      # c: Center of layers (pixel)
        p_f = self.p_r * df                # p_f: H x W x 3, focus positions of retinal pixels on focus plane
        rot_forward = glm.dvec3(glm.tan(glm.radians(glm.dvec2(gaze[1], -gaze[0]))), 1)
        rot_mat = torch.from_numpy(np.array(
            glm.dmat3(glm.lookAtLH(glm.dvec3(), rot_forward, glm.dvec3(0, 1, 0)))))
        rot_mat = rot_mat.float()
        u_rot = torch.mm(self.u, rot_mat)
        v_rot = torch.matmul(p_f, rot_mat).unsqueeze(2).expand(
            -1, -1, self.u.size()[0], -1) - u_rot # v_rot: H x W x M x 3, rotated rays' direction vector
        v_rot.div_(v_rot[:, :, :, 2].unsqueeze(3))             # make z = 1 for each direction vector in v_rot
        
        for i in range(0, self.conf.GetNLayers()):
            dp_i = self.conf.GetLayerSize(i)[0] / self.conf.layer_res[0] # dp_i: Pixel size of layer i
            d_i = self.conf.d_layer[i]                                        # d_i: Distance of layer i
            k = (d_i - u_rot[:, 2]).unsqueeze(1)
            pi_r = (u_rot[:, 0:2] + v_rot[:, :, :, 0:2] * k) / dp_i      # pi_r: H x W x M x 2, rays' pixel coord on layer i
            Phi[i, :, :, :, :] = torch.floor(pi_r + c)
        mask[:, :, :, :, 0] = ((Phi[:, :, :, :, 0] >= 0) & (Phi[:, :, :, :, 0] < self.conf.layer_res[0])).float()
        mask[:, :, :, :, 1] = ((Phi[:, :, :, :, 1] >= 0) & (Phi[:, :, :, :, 1] < self.conf.layer_res[1])).float()
        Phi[:, :, :, :, 0].clamp_(0, self.conf.layer_res[0] - 1)
        Phi[:, :, :, :, 1].clamp_(0, self.conf.layer_res[1] - 1)
        retinal_mask = mask.prod(0).prod(2).prod(2)
        return [ Phi, retinal_mask ]
    
    def GenRetinalFromLayers(self, layers, Phi):
        '''
        Generate retinal image from layers, using precalculated mapping matrix
        
        Parameters
        --------
        layers - 3N x H_l x W_l tensor, stacked layer images, with 3 channels in each layer
        
        Returns
        --------
        3 x H_r x W_r tensor, 3 channels retinal image
        H_r x W_r tensor, retinal image mask, indicates pixels valid or not
        
        '''
        # FOR GRAYSCALE 1 FOR RGB 3
        mapped_layers = torch.empty(self.N, 3, self.D_r[0], self.D_r[1], self.M) # 2 x 3 x 480 x 640 x 41
        # print("mapped_layers:",mapped_layers.shape)
        for i in range(0, Phi.size()[0]):
            # print("gather layers:",layers[(i * 3) : (i * 3 + 3),Phi[i, :, :, :, 0],Phi[i, :, :, :, 1]].shape)
            mapped_layers[i, :, :, :, :] = layers[(i * 3) : (i * 3 + 3),
                                                    Phi[i, :, :, :, 0],
                                                    Phi[i, :, :, :, 1]]
        # print("mapped_layers:",mapped_layers.shape)
        retinal = mapped_layers.prod(0).sum(3).div(Phi.size()[3])
        # print("retinal:",retinal.shape)
        return retinal
        
    def GenFoveaLayers(self, retinal, retinal_mask):
        '''
        Generate foveated layers and corresponding masks
        
        Parameters
        --------
        retinal      - Retinal image generated by GenRetinalFromLayers()
        retinal_mask - Mask of retinal image, also generated by GenRetinalFromLayers()
        
        Returns
        --------
        fovea_layers      - list of foveated layers
        fovea_layer_masks - list of mask images, corresponding to foveated layers
        '''
        fovea_layers = []
        fovea_layer_masks = []
        fov = self.conf.eye_fovea_angles[-1]
        retinal_res = int(self.conf.retinal_res[0])
        for i in range(0, len(self.conf.eye_fovea_angles)):
            angle = self.conf.eye_fovea_angles[i]
            k = self.conf.eye_fovea_downsamples[i]
            roi_size = int(np.ceil(retinal_res * angle / fov))
            roi_offset = int((retinal_res - roi_size) / 2)
            roi_img = retinal[:, roi_offset:(roi_offset + roi_size), roi_offset:(roi_offset + roi_size)]
            roi_mask = retinal_mask[roi_offset:(roi_offset + roi_size), roi_offset:(roi_offset + roi_size)]
            if k == 1:
                fovea_layers.append(roi_img)
                fovea_layer_masks.append(roi_mask)
            else:
                fovea_layers.append(torch.nn.functional.avg_pool2d(roi_img.unsqueeze(0), k).squeeze(0))
                fovea_layer_masks.append(1 - torch.nn.functional.max_pool2d((1 - roi_mask).unsqueeze(0), k).squeeze(0))
        return [ fovea_layers, fovea_layer_masks ]