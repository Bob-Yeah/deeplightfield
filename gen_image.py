import matplotlib.pyplot as plt
import numpy as np
import torch
import glm
import time
import util

def RandomGenSamplesInPupil(pupil_size, n_samples):
    '''
    Random sample n_samples positions in pupil region
    
    Parameters
    --------
    conf      - multi-layers' parameters configuration
    n_samples - number of samples to generate
    
    Returns
    --------
    a n_samples x 3 tensor with 3D sample position in each row
    '''
    samples = torch.empty(n_samples, 3)
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
    a n_samples x 3 tensor with 3D sample position in each row
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
    def __init__(self, conf):
        '''
        Initialize retinal generator instance

        Parameters
        --------
        conf - multi-layers' parameters configuration
        u    - a M x 3 tensor stores M sample positions in pupil
        '''
        self.conf = conf
        self.u = GenSamplesInPupil(conf.pupil_size, 5)
        # self.u = u.to(cuda_dev)
        # self.u = u # M x 3 M sample positions 
        self.D_r = conf.retinal_res # retinal res 480 x 640 
        self.N = conf.GetNLayers() # 2 
        self.M = self.u.size()[0] # samples
        # p_rx, p_ry = torch.meshgrid(torch.tensor(range(0, self.D_r[0])),
        #                             torch.tensor(range(0, self.D_r[1])))
        # self.p_r = torch.cat([
        #     ((torch.stack([p_rx, p_ry], 2) + 0.5) / self.D_r - 0.5) * conf.GetEyeViewportSize(), # 眼球视野
        #     torch.ones(self.D_r[0], self.D_r[1], 1)
        # ], 2)

        self.p_r = torch.cat([
            ((util.MeshGrid(self.D_r) + 0.5) / self.D_r - 0.5) * conf.GetEyeViewportSize(),
            torch.ones(self.D_r[0], self.D_r[1], 1)
        ], 2)

        # self.Phi = torch.empty(N, D_r[0], D_r[1], M, 2, device=cuda_dev, dtype=torch.long)
        # self.mask = torch.empty(self.N, self.D_r[0], self.D_r[1], self.M, 2, dtype=torch.float) # 2 x 480 x 640 x 41 x 2
        
    def CalculateRetinal2LayerMappings(self, position, gaze_dir, df):
        '''
        Calculate the mapping matrix from retinal to layers.

        Parameters
        --------
        position - 1 x 3 tensor, eye's position
        gaze_dir - 1 x 2 tensor, gaze forward vector (with z normalized)
        df       - focus distance

        Returns
        --------
        phi             - N x H_r x W_r x M x 2, retinal to layers mapping, N is number of layers
        phi_invalid     - N x H_r x W_r x M x 1, indicates invalid (out-of-range) mapping
        retinal_invalid - 1 x H_r x W_r, indicates invalid pixels in retinal image
        '''
        D = self.conf.layer_res 
        c = torch.tensor([ D[1] / 2, D[0] / 2 ])     # c: Center of layers (pixel)

        D_r = self.conf.retinal_res        # D_r: Resolution of retinal 480 640
        V = self.conf.GetEyeViewportSize() # V: Viewport size of eye 
        p_f = self.p_r * df                # p_f: H x W x 3, focus positions of retinal pixels on focus plane
        
        # Calculate transformation from eye to display
        gvec_lookat = glm.dvec3(gaze_dir[0], -gaze_dir[1], 1)
        gmat_eye = glm.inverse(glm.lookAtLH(glm.dvec3(), gvec_lookat, glm.dvec3(0, 1, 0)))
        eye_rot = util.Glm2Tensor(glm.dmat3(gmat_eye))
        eye_center = torch.tensor([ position[0], -position[1], position[2] ])

        u_rot = torch.mm(self.u, eye_rot)
        v_rot = torch.matmul(p_f, eye_rot).unsqueeze(2).expand(
            -1, -1, self.M, -1) - u_rot # v_rot: H x W x M x 3, rotated rays' direction vector
        u_rot.add_(eye_center)                            # translate by eye's center
        v_rot = v_rot.div(v_rot[:, :, :, 2].unsqueeze(3)) # make z = 1 for each direction vector in v_rot
        
        phi = torch.empty(self.N, self.D_r[0], self.D_r[1], self.M, 2, dtype=torch.long)

        for i in range(0, self.N):
            dp_i = self.conf.GetPixelSizeOfLayer(i)     # dp_i: Pixel size of layer i
            d_i = self.conf.d_layer[i]                  # d_i: Distance of layer i
            k = (d_i - u_rot[:, 2]).unsqueeze(1)
            pi_r = (u_rot[:, 0:2] + v_rot[:, :, :, 0:2] * k) / dp_i      # pi_r: H x W x M x 2, rays' pixel coord on layer i
            phi[i, :, :, :, :] = torch.floor(pi_r + c)
        
        # Calculate invalid mask (out-of-range elements in phi) and reduced to retinal
        phi_invalid = (phi[:, :, :, :, 0] < 0) | (phi[:, :, :, :, 0] >= D[1]) | \
                       (phi[:, :, :, :, 1] < 0) | (phi[:, :, :, :, 1] >= D[0])
        phi_invalid = phi_invalid.unsqueeze(4)
        # print("phi_invalid:",phi_invalid.shape) 
        retinal_invalid = phi_invalid.amax((0, 3)).squeeze().unsqueeze(0)
        # print("retinal_invalid:",retinal_invalid.shape)
        # Fix invalid elements in phi
        phi[phi_invalid.expand(-1, -1, -1, -1, 2)] = 0

        return [ phi, phi_invalid, retinal_invalid  ]
    
    
    def GenRetinalFromLayers(self, layers, Phi):
        '''
        Generate retinal image from layers, using precalculated mapping matrix
        
        Parameters
        --------
        layers       - 3N x H x W, stacked layer images, with 3 channels in each layer
        phi          - N x H_r x W_r x M x 2, retinal to layers mapping, N is number of layers
        
        Returns
        --------
        3 x H_r x W_r, 3 channels retinal image
        '''
        # FOR GRAYSCALE 1 FOR RGB 3
        mapped_layers = torch.empty(self.N, 3, self.D_r[0], self.D_r[1], self.M) # 2 x 3 x 480 x 640 x 41
        # print("mapped_layers:",mapped_layers.shape)
        for i in range(0, Phi.size()[0]):
            # torch.Size([3, 2, 320, 320, 2])
            # print("gather layers:",layers[(i * 3) : (i * 3 + 3),Phi[i, :, :, :, 0],Phi[i, :, :, :, 1]].shape)
            mapped_layers[i, :, :, :, :] = layers[(i * 3) : (i * 3 + 3),
                                                    Phi[i, :, :, :, 1],
                                                    Phi[i, :, :, :, 0]]
        # print("mapped_layers:",mapped_layers.shape)
        retinal = mapped_layers.prod(0).sum(3).div(Phi.size()[3])
        # print("retinal:",retinal.shape)
        return retinal

    def GenRetinalFromLayersBatch(self, layers, Phi):
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
        mapped_layers = torch.empty(layers.size()[0], self.N, 3, self.D_r[0], self.D_r[1], self.M) #BS x Layers x C x H x W x Sample
        
        # truth = torch.empty(layers.size()[0], self.N, 3, self.D_r[0], self.D_r[1], self.M)
        # layers_truth = layers.clone()
        # Phi_truth = Phi.clone()
        layers = torch.stack((layers[:,0:3,:,:],layers[:,3:6,:,:]),dim=1) ## torch.Size([BS, Layer, RGB 3, 320, 320])
        
        # Phi = Phi[:,:,None,:,:,:,:].expand(-1,-1,3,-1,-1,-1,-1)
        # print("mapped_layers:",mapped_layers.shape) #torch.Size([2, 2, 3, 320, 320, 41])
        # print("input layers:",layers.shape) ## torch.Size([2, 2, 3, 320, 320])
        # print("input Phi:",Phi.shape) #torch.Size([2, 2, 320, 320, 41, 2])
        
        # #没优化

        # for i in range(0, Phi_truth.size()[0]):
        #     for j in range(0, Phi_truth.size()[1]):
        #         truth[i, j, :, :, :, :] = layers_truth[i, (j * 3) : (j * 3 + 3),
        #                                                 Phi_truth[i, j, :, :, :, 0],
        #                                                 Phi_truth[i, j, :, :, :, 1]]

        #优化2
        # start = time.time() 
        mapped_layers_op1 = mapped_layers.reshape(-1,
                mapped_layers.shape[2],mapped_layers.shape[3],mapped_layers.shape[4],mapped_layers.shape[5])
                # BatchSizexLayer Channel 3 320 320 41
        layers_op1 = layers.reshape(-1,layers.shape[2],layers.shape[3],layers.shape[4]) # 2x2 3 320 320
        Phi_op1 = Phi.reshape(-1,Phi.shape[2],Phi.shape[3],Phi.shape[4],Phi.shape[5]) # 2x2 320 320 41 2
        x = Phi_op1[:,:,:,:,0] # 2x2 320 320 41
        y = Phi_op1[:,:,:,:,1] # 2x2 320 320 41
        # print("reshape:",time.time() - start)

        # start = time.time()
        mapped_layers_op1 = layers_op1[torch.arange(layers_op1.shape[0])[:, None, None, None], :, y, x] # x,y 切换
        #2x2 320 320 41 3
        # print("mapping one step:",time.time() - start)
        
        # print("mapped_layers:",mapped_layers_op1.shape) # torch.Size([4, 3, 320, 320, 41])
        # start = time.time()
        mapped_layers_op1 = mapped_layers_op1.permute(0,4,1,2,3)
        mapped_layers = mapped_layers_op1.reshape(mapped_layers.shape[0],mapped_layers.shape[1],
                    mapped_layers.shape[2],mapped_layers.shape[3],mapped_layers.shape[4],mapped_layers.shape[5])
        # print("reshape end:",time.time() - start)

        # print("test:")
        # print((truth.cpu() == mapped_layers.cpu()).all())
        #优化1
        # start = time.time()
        # mapped_layers_op1 = mapped_layers.reshape(-1,
        #         mapped_layers.shape[2],mapped_layers.shape[3],mapped_layers.shape[4],mapped_layers.shape[5])
        # layers_op1 = layers.reshape(-1,layers.shape[2],layers.shape[3],layers.shape[4])
        # Phi_op1 = Phi.reshape(-1,Phi.shape[2],Phi.shape[3],Phi.shape[4],Phi.shape[5])
        # print("reshape:",time.time() - start)


        # for i in range(0, Phi_op1.size()[0]):
        #     start = time.time()
        #     mapped_layers_op1[i, :, :, :, :] = layers_op1[i,:,
        #                                             Phi_op1[i, :, :, :, 0],
        #                                             Phi_op1[i, :, :, :, 1]]
        #     print("mapping one step:",time.time() - start)
        # print("mapped_layers:",mapped_layers_op1.shape) # torch.Size([4, 3, 320, 320, 41])
        # start = time.time()
        # mapped_layers = mapped_layers_op1.reshape(mapped_layers.shape[0],mapped_layers.shape[1],
        #             mapped_layers.shape[2],mapped_layers.shape[3],mapped_layers.shape[4],mapped_layers.shape[5])
        # print("reshape end:",time.time() - start)

        # print("mapped_layers:",mapped_layers.shape) # torch.Size([2, 2, 3, 320, 320, 41])
        retinal = mapped_layers.prod(1).sum(4).div(Phi.size()[4])
        # print("retinal:",retinal.shape) # torch.Size([BatchSize, 3, 320, 320])
        return retinal
        
    ## TO BE CHECK
    def GenFoveaLayers(self, b_retinal, is_mask):
        '''
        Generate foveated layers for retinal images or masks
        
        Parameters
        --------
        b_retinal - B x C x H_r x W_r, Batch of retinal images/masks
        is_mask   - Whether b_retinal is masks or images
        
        Returns
        --------
        b_fovea_layers - N_f x (B x C x H[f] x W[f]) list of batch of foveated layers
        '''
        b_fovea_layers = []
        for i in range(0, len(self.conf.eye_fovea_angles)):
            k = self.conf.eye_fovea_downsamples[i]
            region = self.conf.GetRegionOfFoveaLayer(i)
            b_roi = b_retinal[:, :, region, region]
            if k == 1:
                b_fovea_layers.append(b_roi)
            elif is_mask:
                b_fovea_layers.append(torch.nn.functional.max_pool2d(b_roi.to(torch.float), k).to(torch.bool))
            else:
                b_fovea_layers.append(torch.nn.functional.avg_pool2d(b_roi, k))
        return b_fovea_layers
        # fovea_layers = []
        # fovea_layer_masks = []
        # fov = self.conf.eye_fovea_angles[-1]
        # retinal_res = int(self.conf.retinal_res[0])
        # for i in range(0, len(self.conf.eye_fovea_angles)):
        #     angle = self.conf.eye_fovea_angles[i]
        #     k = self.conf.eye_fovea_downsamples[i]
        #     roi_size = int(np.ceil(retinal_res * angle / fov))
        #     roi_offset = int((retinal_res - roi_size) / 2)
        #     roi_img = retinal[:, roi_offset:(roi_offset + roi_size), roi_offset:(roi_offset + roi_size)]
        #     roi_mask = retinal_mask[roi_offset:(roi_offset + roi_size), roi_offset:(roi_offset + roi_size)]
        #     if k == 1:
        #         fovea_layers.append(roi_img)
        #         fovea_layer_masks.append(roi_mask)
        #     else:
        #         fovea_layers.append(torch.nn.functional.avg_pool2d(roi_img.unsqueeze(0), k).squeeze(0))
        #         fovea_layer_masks.append(1 - torch.nn.functional.max_pool2d((1 - roi_mask).unsqueeze(0), k).squeeze(0))
        # return [ fovea_layers, fovea_layer_masks ]

    ## TO BE CHECK
    def GenFoveaLayersBatch(self, retinal, retinal_mask):
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
        # print("fov:",fov)
        retinal_res = int(self.conf.retinal_res[0])
        # print("retinal_res:",retinal_res)
        # print("len(self.conf.eye_fovea_angles):",len(self.conf.eye_fovea_angles))
        for i in range(0, len(self.conf.eye_fovea_angles)):
            angle = self.conf.eye_fovea_angles[i]
            k = self.conf.eye_fovea_downsamples[i]
            roi_size = int(np.ceil(retinal_res * angle / fov))
            roi_offset = int((retinal_res - roi_size) / 2)
            # [2, 3, 320, 320]
            roi_img = retinal[:, :, roi_offset:(roi_offset + roi_size), roi_offset:(roi_offset + roi_size)]
            # print("roi_img:",roi_img.shape)
            # [2, 320, 320]
            roi_mask = retinal_mask[:, roi_offset:(roi_offset + roi_size), roi_offset:(roi_offset + roi_size)]
            # print("roi_mask:",roi_mask.shape)
            if k == 1:
                fovea_layers.append(roi_img)
                fovea_layer_masks.append(roi_mask)
            else:
                fovea_layers.append(torch.nn.functional.avg_pool2d(roi_img, k))
                fovea_layer_masks.append(1 - torch.nn.functional.max_pool2d((1 - roi_mask), k))
        return [ fovea_layers, fovea_layer_masks ]
    
    ## TO BE CHECK
    def GenFoveaRetinal(self, b_fovea_layers):
        '''
        Generate foveated retinal image by blending fovea layers
        **Note: current implementation only support two fovea layers**
        
        Parameters
        --------
        b_fovea_layers - N_f x (B x 3 x H[f] x W[f]), list of batch of (masked) foveated layers
        
        Returns
        --------
        B x 3 x H_r x W_r, batch of foveated retinal images
        '''
        b_fovea_retinal = torch.nn.functional.interpolate(b_fovea_layers[1],
            scale_factor=self.conf.eye_fovea_downsamples[1],
            mode='bilinear', align_corners=False)
        region = self.conf.GetRegionOfFoveaLayer(0)
        blend = self.conf.eye_fovea_blend[0]
        b_roi = b_fovea_retinal[:, :, region, region]
        b_roi.mul_(1 - blend).add_(b_fovea_layers[0] * blend)
        return b_fovea_retinal
