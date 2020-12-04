import torch
from gen_image import *
class Conf(object):
    def __init__(self):
        self.pupil_size = 0.02
        self.retinal_res = torch.tensor([ 320, 320 ])
        self.layer_res = torch.tensor([ 320, 320 ])
        self.layer_hfov = 90                  # layers' horizontal FOV
        self.eye_hfov = 85                    # eye's horizontal FOV (ignored in foveated rendering)
        self.eye_enable_fovea = True          # enable foveated rendering
        self.eye_fovea_angles = [ 40, 80 ]    # eye's foveation layers' angles
        self.eye_fovea_downsamples = [ 1, 2 ] # eye's foveation layers' downsamples
        self.d_layer = [ 1, 3 ]               # layers' distance
        
    def GetNLayers(self):
        return len(self.d_layer)
    
    def GetLayerSize(self, i):
        w = Fov2Length(self.layer_hfov)
        h = w * self.layer_res[0] / self.layer_res[1]
        return torch.tensor([ h, w ]) * self.d_layer[i]

    def GetEyeViewportSize(self):
        fov = self.eye_fovea_angles[-1] if self.eye_enable_fovea else self.eye_hfov
        w = Fov2Length(fov)
        h = w * self.retinal_res[0] / self.retinal_res[1]
        return torch.tensor([ h, w ])