import torch
import util
import numpy as np
class Conf(object):
    def __init__(self):
        self.pupil_size = 0.02
        self.retinal_res = torch.tensor([ 320, 320 ])
        self.layer_res = torch.tensor([ 320, 320 ])
        self.layer_hfov = 90                  # layers' horizontal FOV
        self.eye_hfov = 80                    # eye's horizontal FOV (ignored in foveated rendering)
        self.eye_enable_fovea = False          # enable foveated rendering
        self.eye_fovea_angles = [ 40, 80 ]    # eye's foveation layers' angles
        self.eye_fovea_downsamples = [ 1, 2 ] # eye's foveation layers' downsamples
        self.d_layer = [ 1, 3 ]               # layers' distance
        self.eye_fovea_blend = [ self._GenFoveaLayerBlend(0) ]
                                                # blend maps of fovea layers
        self.light_field_dim = 5
    def GetNLayers(self):
        return len(self.d_layer)
    
    def GetLayerSize(self, i):
        w = util.Fov2Length(self.layer_hfov)
        h = w * self.layer_res[0] / self.layer_res[1]
        return torch.tensor([ h, w ]) * self.d_layer[i]

    def GetPixelSizeOfLayer(self, i):
        '''
        Get pixel size of layer i
        '''
        return util.Fov2Length(self.layer_hfov) * self.d_layer[i] / self.layer_res[0]

    def GetEyeViewportSize(self):
        fov = self.eye_fovea_angles[-1] if self.eye_enable_fovea else self.eye_hfov
        w = util.Fov2Length(fov)
        h = w * self.retinal_res[0] / self.retinal_res[1]
        return torch.tensor([ h, w ])

    def GetRegionOfFoveaLayer(self, i):
        '''
        Get region of fovea layer i in retinal image
        
        Returns
        --------
        slice object stores the start and end of region
        '''
        roi_size = int(np.ceil(self.retinal_res[0] * self.eye_fovea_angles[i] / self.eye_fovea_angles[-1]))
        roi_offset = int((self.retinal_res[0] - roi_size) / 2)
        return slice(roi_offset, roi_offset + roi_size)
    
    def _GenFoveaLayerBlend(self, i):
        '''
        Generate blend map for fovea layer i
        
        Parameters
        --------
        i - index of fovea layer
        
        Returns
        --------
        H[i] x W[i], blend map
        
        '''
        region = self.GetRegionOfFoveaLayer(i)
        width = region.stop - region.start
        R = width / 2
        p = util.MeshGrid([ width, width ])
        r = torch.linalg.norm(p - R, 2, dim=2, keepdim=False)
        return util.SmoothStep(R, R * 0.6, r)
