import matplotlib.pyplot as plt
import torch
import util
import numpy as np
def FlowMap(b_last_frame, b_map):
    '''
    Map images using the flow data.
    
    Parameters
    --------
    b_last_frame - B x 3 x H x W tensor, batch of images
    b_map - B x H x W x 2, batch of map data records pixel coords in last frames
    
    Returns
    --------
    B x 3 x H x W tensor, batch of images mapped by flow data
    '''
    return torch.nn.functional.grid_sample(b_last_frame, b_map, align_corners=False)
    
class Flow(object):
    '''
    Class representating optical flow
    
    Properties
    --------
    b_data         - B x H x W x 2, batch of flow data
    b_map          - B x H x W x 2, batch of map data records pixel coords in last frames
    b_invalid_mask - B x H x W, batch of masks, indicate invalid elements in corresponding flow data
    '''
    def Load(paths):
        '''
        Create a Flow instance using a batch of encoded data images loaded from paths
        
        Parameters
        --------
        paths - list of encoded data image paths
        
        Returns
        --------
        Flow instance
        '''
        b_encoded_image = util.ReadImageTensor(paths, rgb_only=False, permute=False, batch_dim=True)
        return Flow(b_encoded_image)

    def __init__(self, b_encoded_image):
        '''
        Initialize a Flow instance from a batch of encoded data images
        
        Parameters
        --------
        b_encoded_image - batch of encoded data images
        '''
        b_encoded_image = b_encoded_image.mul(255)
        # print("b_encoded_image:",b_encoded_image.shape)
        self.b_invalid_mask = (b_encoded_image[:, :, :, 0] == 255)
        self.b_data = (b_encoded_image[:, :, :, 0:2] / 254 + b_encoded_image[:, :, :, 2:4] - 127) / 127
        self.b_data[:, :, :, 1] = -self.b_data[:, :, :, 1]
        D = self.b_data.size()
        grid = util.MeshGrid((D[1], D[2]), True)
        self.b_map = (grid - self.b_data - 0.5) * 2
        self.b_map[self.b_invalid_mask] = torch.tensor([ -2.0, -2.0 ])
    
    def getMap(self):
        return self.b_map

    def Visualize(self, scale_factor = 1):
        '''
        Visualize the flow data by "color wheel".
        
        Parameters
        --------
        scale_factor - scale factor of flow data to visualize, default is 1
        
        Returns
        --------
        B x 3 x H x W tensor, visualization of flow data
        '''
        try:
            Flow.b_color_wheel
        except AttributeError:
            Flow.b_color_wheel = util.ReadImageTensor('color_wheel.png')
        return torch.nn.functional.grid_sample(Flow.b_color_wheel.expand(self.b_data.size()[0], -1, -1, -1),
            (self.b_data * scale_factor), align_corners=False)