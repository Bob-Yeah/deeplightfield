import numpy as np
import torch
import matplotlib.pyplot as plt
import glm

gvec_type = [ glm.dvec1, glm.dvec2, glm.dvec3, glm.dvec4 ]
gmat_type = [ [ glm.dmat2, glm.dmat2x3, glm.dmat2x4 ], 
              [ glm.dmat3x2, glm.dmat3, glm.dmat3x4 ],
              [ glm.dmat4x2, glm.dmat4x3, glm.dmat4 ] ]

def Fov2Length(angle):
    return np.tan(angle * np.pi / 360) * 2

def SmoothStep(x0, x1, x):
    y = torch.clamp((x - x0) / (x1 - x0), 0, 1)
    return y * y * (3 - 2 * y)

def MatImg2Tensor(img, permute = True, batch_dim = True):
    batch_input = len(img.shape) == 4
    if permute:
        t = torch.from_numpy(np.transpose(img,
            [0, 3, 1, 2] if batch_input else [2, 0, 1]))
    else:
        t = torch.from_numpy(img)
    if not batch_input and batch_dim:
        t = t.unsqueeze(0)
    return t

def MatImg2Numpy(img, permute = True, batch_dim = True):
    batch_input = len(img.shape) == 4
    if permute:
        t = np.transpose(img, [0, 3, 1, 2] if batch_input else [2, 0, 1])
    else:
        t = img
    if not batch_input and batch_dim:
        t = t.unsqueeze(0)
    return t
    
def Tensor2MatImg(t):
    img = t.squeeze().cpu().numpy()
    batch_input = len(img.shape) == 4
    if t.size()[batch_input] <= 4:
        return np.transpose(img, [0, 2, 3, 1] if batch_input else [1, 2, 0])
    return img

def ReadImageTensor(path, permute = True, rgb_only = True, batch_dim = True):
    channels = 3 if rgb_only else 4
    if isinstance(path,list):
        first_image = plt.imread(path[0])[:, :, 0:channels]
        b_image = np.empty((len(path), first_image.shape[0], first_image.shape[1], channels), dtype=np.float32)
        b_image[0] = first_image
        for i in range(1, len(path)):
            b_image[i] = plt.imread(path[i])[:, :, 0:channels]
        return MatImg2Tensor(b_image, permute)
    return MatImg2Tensor(plt.imread(path)[:, :, 0:channels], permute, batch_dim)

def ReadImageNumpyArray(path, permute = True, rgb_only = True, batch_dim = True):
    channels = 3 if rgb_only else 4
    if isinstance(path,list):
        first_image = plt.imread(path[0])[:, :, 0:channels]
        b_image = np.empty((len(path), first_image.shape[0], first_image.shape[1], channels), dtype=np.float32)
        b_image[0] = first_image
        for i in range(1, len(path)):
            b_image[i] = plt.imread(path[i])[:, :, 0:channels]
        return MatImg2Numpy(b_image, permute)
    return MatImg2Numpy(plt.imread(path)[:, :, 0:channels], permute, batch_dim)

def WriteImageTensor(t, path):
    image = Tensor2MatImg(t)
    if isinstance(path,list):
        if len(image.shape) != 4 or image.shape[0] != len(path):
            raise ValueError
        for i in range(len(path)):
            plt.imsave(path[i], image[i])
    else:
        if len(image.shape) == 4 and image.shape[0] != 1:
            raise ValueError
        plt.imsave(path, image)
    
def PlotImageTensor(t):
    plt.imshow(Tensor2MatImg(t))
    
def Tensor2Glm(t):
    t = t.squeeze()
    size = t.size()
    if len(size) == 1:
        if size[0] <= 0 or size[0] > 4:
            raise ValueError
        return gvec_type[size[0] - 1](t.cpu().numpy())
    if len(size) == 2:
        if size[0] <= 1 or size[0] > 4 or size[1] <= 1 or size[1] > 4:
            raise ValueError
        return gmat_type[size[1] - 2][size[0] - 2](t.cpu().numpy())
    raise ValueError

def Glm2Tensor(val):
    return torch.from_numpy(np.array(val))

def MeshGrid(size, normalize=False):
    y,x = torch.meshgrid(torch.tensor(range(size[0])),
                          torch.tensor(range(size[1])))
    if normalize:
        return torch.stack([x / (size[1] - 1.), y / (size[0] - 1.)], 2)
    return torch.stack([x, y], 2)