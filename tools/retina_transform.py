import cv2
import numpy as np
import os
from utils.constants import *


def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    
    
    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids

def foveat_img(im, fixs):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    sigma=0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    
    # compute coef
    p = 7.5
    k = 3
    alpha = 2.5

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega>1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + TINY_FLOAT))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    print('num of full-res pixel', np.sum(Ms[0] == 1))
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov

import json

def load_centers(data_desc_file):
    with open(data_desc_file, 'r', encoding='utf-8') as file:
        data_desc = json.loads(file.read())
        return data_desc['gaze_centers']

hint = cv2.imread(sys.path[0] + '/fovea_hint.png', rgb_only=False)

def add_hint(img, center):
    fovea_origin = (
        int(center[0]) + res_full[1] // 2 - hint.size(-1) // 2,
        int(center[1]) + res_full[0] // 2 - hint.size(-2) // 2
    )
    fovea_region = (
        slice(fovea_origin[1], fovea_origin[1] + hint.size(-2)),
        slice(fovea_origin[0], fovea_origin[0] + hint.size(-1)),
        ...
    )
    img[fovea_region] = img[fovea_region] * (1 - hint[:, 3:]) + \
        hint[:, :3] * hint[:, 3:]

if __name__ == "__main__":
    datadir = 'D:\\deeplightfield\\data\\hmd_gas\\'
    centers = load_centers(datadir + 'left.json')
    image_paths = [
        datadir + 'left\\view_%04d.png' % i
        for i in range(len(centers))
    ]
    if not os.path.exists(datadir + 'left_foveated'):
        os.makedirs(datadir + 'left_foveated')

    i = 0
    for c, path in zip(centers, image_paths):
        # Left eye
        im = cv2.imread(path)
        xc, yc = im.shape[1] // 2 + c[0], im.shape[0] // 2 + c[1]
        im = foveat_img(im, [(xc, yc)])
        add_hint(im, c)
        cv2.imwrite(os.path.join(datadir, 'left_foveated', 'view_%04d.png' % i), im)
        print('Frame %04d saved' % i)
        i += 1
