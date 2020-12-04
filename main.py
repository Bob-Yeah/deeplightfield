import torch
import argparse
import os
import glob
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.utils.data import DataLoader 
from torch.autograd import Variable

import cv2
from gen_image import *
import json
from ssim import *
from perc_loss import * 
from conf import Conf

from model.baseline import *

import torch.autograd.profiler as profiler
# param
BATCH_SIZE = 2
NUM_EPOCH = 300

INTERLEAVE_RATE = 2

IM_H = 320
IM_W = 320

Retinal_IM_H = 320
Retinal_IM_W = 320

N = 9 # number of input light field stack
M = 2 # number of display layers

DATA_FILE = "/home/yejiannan/Project/LightField/data/gaze_fovea"
DATA_JSON = "/home/yejiannan/Project/LightField/data/data_gaze_fovea_seq.json"
DATA_VAL_JSON = "/home/yejiannan/Project/LightField/data/data_gaze_fovea_val.json"
OUTPUT_DIR = "/home/yejiannan/Project/LightField/outputE/gaze_fovea_seq"


OUT_CHANNELS_RB = 128
KERNEL_SIZE_RB = 3
KERNEL_SIZE = 3

LAST_LAYER_CHANNELS = 6 * INTERLEAVE_RATE**2
FIRSST_LAYER_CHANNELS = 27 * INTERLEAVE_RATE**2

class lightFieldDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path, file_json, transforms=None):
        self.file_dir_path = file_dir_path
        self.transforms = transforms
        # self.datum_list = glob.glob(os.path.join(file_dir_path,"*"))
        with open(file_json, encoding='utf-8') as file:
            self.dataset_desc = json.loads(file.read())

    def __len__(self):
        return len(self.dataset_desc["focaldepth"])

    def __getitem__(self, idx):
        lightfield_images, gt, gt2, fd, gazeX, gazeY, sample_idx = self.get_datum(idx)
        if self.transforms:
            lightfield_images = self.transforms(lightfield_images)
        # print(lightfield_images.shape,gt.shape,fd,gazeX,gazeY,sample_idx)
        return (lightfield_images, gt, gt2, fd, gazeX, gazeY, sample_idx)

    def get_datum(self, idx):
        lf_image_paths = os.path.join(DATA_FILE, self.dataset_desc["train"][idx])
        # print(lf_image_paths)
        fd_gt_path = os.path.join(DATA_FILE, self.dataset_desc["gt"][idx])
        fd_gt_path2 = os.path.join(DATA_FILE, self.dataset_desc["gt2"][idx])
        # print(fd_gt_path)
        lf_images = []
        lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)

        ## IF GrayScale
        # lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        # lf_image_big = np.expand_dims(lf_image_big, axis=-1)
        # print(lf_image_big.shape)

        for i in range(9):
            lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:3]
            ## IF GrayScale
            # lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:1]
            # print(lf_image.shape)
            lf_images.append(lf_image)
        gt = cv2.imread(fd_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        gt2 = cv2.imread(fd_gt_path2, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        gt2 = cv2.cvtColor(gt2,cv2.COLOR_BGR2RGB)
        ## IF GrayScale
        # gt = cv2.imread(fd_gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        # gt = np.expand_dims(gt, axis=-1)

        fd = self.dataset_desc["focaldepth"][idx]
        gazeX = self.dataset_desc["gazeX"][idx]
        gazeY = self.dataset_desc["gazeY"][idx]
        sample_idx = self.dataset_desc["idx"][idx]
        return np.asarray(lf_images),gt,gt2,fd,gazeX,gazeY,sample_idx

class lightFieldValDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path, file_json, transforms=None):
        self.file_dir_path = file_dir_path
        self.transforms = transforms
        # self.datum_list = glob.glob(os.path.join(file_dir_path,"*"))
        with open(file_json, encoding='utf-8') as file:
            self.dataset_desc = json.loads(file.read())

    def __len__(self):
        return len(self.dataset_desc["focaldepth"])

    def __getitem__(self, idx):
        lightfield_images, fd, gazeX, gazeY, sample_idx = self.get_datum(idx)
        if self.transforms:
            lightfield_images = self.transforms(lightfield_images)
        # print(lightfield_images.shape,gt.shape,fd,gazeX,gazeY,sample_idx)
        return (lightfield_images, fd, gazeX, gazeY, sample_idx)

    def get_datum(self, idx):
        lf_image_paths = os.path.join(DATA_FILE, self.dataset_desc["train"][idx])
        # print(fd_gt_path)
        lf_images = []
        lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)

        ## IF GrayScale
        # lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        # lf_image_big = np.expand_dims(lf_image_big, axis=-1)
        # print(lf_image_big.shape)

        for i in range(9):
            lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:3]
            ## IF GrayScale
            # lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:1]
            # print(lf_image.shape)
            lf_images.append(lf_image)
        ## IF GrayScale
        # gt = cv2.imread(fd_gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        # gt = np.expand_dims(gt, axis=-1)

        fd = self.dataset_desc["focaldepth"][idx]
        gazeX = self.dataset_desc["gazeX"][idx]
        gazeY = self.dataset_desc["gazeY"][idx]
        sample_idx = self.dataset_desc["idx"][idx]
        return np.asarray(lf_images),fd,gazeX,gazeY,sample_idx

class lightFieldSeqDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path, file_json, transforms=None):
        self.file_dir_path = file_dir_path
        self.transforms = transforms
        with open(file_json, encoding='utf-8') as file:
            self.dataset_desc = json.loads(file.read())

    def __len__(self):
        return len(self.dataset_desc["seq"])

    def __getitem__(self, idx):
        lightfield_images, gt, gt2, fd, gazeX, gazeY, sample_idx = self.get_datum(idx)
        fd = fd.astype(np.float32)
        gazeX = gazeX.astype(np.float32)
        gazeY = gazeY.astype(np.float32)
        sample_idx = sample_idx.astype(np.int64)
        # print(fd)
        # print(gazeX)
        # print(gazeY)
        # print(sample_idx)

        # print(lightfield_images.dtype,gt.dtype, gt2.dtype, fd.dtype, gazeX.dtype, gazeY.dtype, sample_idx.dtype, delta.dtype)
        # print(lightfield_images.shape,gt.shape, gt2.shape, fd.shape, gazeX.shape, gazeY.shape, sample_idx.shape, delta.shape)
        if self.transforms:
            lightfield_images = self.transforms(lightfield_images)
        return (lightfield_images, gt, gt2, fd, gazeX, gazeY, sample_idx)

    def get_datum(self, idx):
        indices = self.dataset_desc["seq"][idx]
        # print("indices:",indices)
        lf_images = []
        fd = []
        gazeX = []
        gazeY = []
        sample_idx = []
        gt = []
        gt2 = []
        for i in range(len(indices)):
            lf_image_paths = os.path.join(DATA_FILE, self.dataset_desc["train"][indices[i]])
            fd_gt_path = os.path.join(DATA_FILE, self.dataset_desc["gt"][indices[i]])
            fd_gt_path2 = os.path.join(DATA_FILE, self.dataset_desc["gt2"][indices[i]])
            lf_image_one_sample = []
            lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)

            for j in range(9):
                lf_image = lf_image_big[j//3*IM_H:j//3*IM_H+IM_H,j%3*IM_W:j%3*IM_W+IM_W,0:3]
                ## IF GrayScale
                # lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:1]
                # print(lf_image.shape)
                lf_image_one_sample.append(lf_image)

            gt_i = cv2.imread(fd_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            gt.append(cv2.cvtColor(gt_i,cv2.COLOR_BGR2RGB))
            gt2_i = cv2.imread(fd_gt_path2, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            gt2.append(cv2.cvtColor(gt2_i,cv2.COLOR_BGR2RGB))

            # print("indices[i]:",indices[i])
            fd.append([self.dataset_desc["focaldepth"][indices[i]]])
            gazeX.append([self.dataset_desc["gazeX"][indices[i]]])
            gazeY.append([self.dataset_desc["gazeY"][indices[i]]])
            sample_idx.append([self.dataset_desc["idx"][indices[i]]])
            lf_images.append(lf_image_one_sample)
        #lf_images: 5,9,320,320

        return np.asarray(lf_images),np.asarray(gt),np.asarray(gt2),np.asarray(fd),np.asarray(gazeX),np.asarray(gazeY),np.asarray(sample_idx)

#### Image Gen
conf = Conf()
u = GenSamplesInPupil(conf.pupil_size, 5)
gen = RetinalGen(conf, u)

def GenRetinalFromLayersBatch(layers, gen, sample_idx, phi_dict, mask_dict):
    # layers: batchsize, 2*color, height, width 
    # Phi:torch.Size([batchsize, 480, 640, 2, 41, 2])
    # df : batchsize,..
    
    #  retinal bs x color x height x width
    retinal = torch.zeros(layers.shape[0], 3, Retinal_IM_H, Retinal_IM_W)
    mask = [] # mask shape 480 x 640
    for i in range(0, layers.size()[0]):
        phi = phi_dict[int(sample_idx[i].data)]
        # print("phi_i:",phi.shape)
        phi = var_or_cuda(phi)
        phi.requires_grad = False
        # print("layers[i]:",layers[i].shape)
        # print("retinal[i]:",retinal[i].shape)
        retinal[i] = gen.GenRetinalFromLayers(layers[i],phi)
        mask.append(mask_dict[int(sample_idx[i].data)])
    retinal = var_or_cuda(retinal)
    mask = torch.stack(mask,dim = 0).unsqueeze(1) # batch x 1 x height x width
    return retinal, mask

def GenRetinalGazeFromLayersBatch(layers, gen, sample_idx, phi_dict, mask_dict):
    # layers: batchsize, 2*color, height, width 
    # Phi:torch.Size([batchsize, 480, 640, 2, 41, 2])
    # df : batchsize,..
    
    #  retinal bs x color x height x width
    retinal_fovea = torch.empty(layers.shape[0], 6, 160, 160)
    mask_fovea = torch.empty(layers.shape[0], 2, 160, 160)
    for i in range(0, layers.size()[0]):
        phi = phi_dict[int(sample_idx[i].data)]
        # print("phi_i:",phi.shape)
        phi = var_or_cuda(phi)
        phi.requires_grad = False
        mask_i = var_or_cuda(mask_dict[int(sample_idx[i].data)])
        mask_i.requires_grad = False
        # print("layers[i]:",layers[i].shape)
        # print("retinal[i]:",retinal[i].shape)
        retinal_i = gen.GenRetinalFromLayers(layers[i],phi)
        fovea_layers, fovea_layer_masks = gen.GenFoveaLayers(retinal_i,mask_i)
        retinal_fovea[i] = torch.cat([fovea_layers[0],fovea_layers[1]],dim=0)
        mask_fovea[i] = torch.stack([fovea_layer_masks[0],fovea_layer_masks[1]],dim=0)
        
    retinal_fovea = var_or_cuda(retinal_fovea)
    mask_fovea = var_or_cuda(mask_fovea) # batch x 2 x height x width
    # mask = torch.stack(mask,dim = 0).unsqueeze(1) 
    return retinal_fovea, mask_fovea

def GenRetinalFromLayersBatch_Online(layers, gen, phi, mask):
    # layers: batchsize, 2*color, height, width 
    # Phi:torch.Size([batchsize, 480, 640, 2, 41, 2])
    # df : batchsize,..
    
    #  retinal bs x color x height x width
    # retinal = torch.zeros(layers.shape[0], 3, Retinal_IM_H, Retinal_IM_W)
    # retinal = var_or_cuda(retinal)
    phi = var_or_cuda(phi)
    phi.requires_grad = False
    retinal = gen.GenRetinalFromLayers(layers[0],phi)
    retinal = var_or_cuda(retinal)
    mask_out = mask.unsqueeze(0).unsqueeze(0)
    # print("maskOUt:",mask_out.shape) # 1,1,240,320
    # mask_out = torch.stack(mask,dim = 0).unsqueeze(1) # batch x 1 x height x width
    return retinal.unsqueeze(0), mask_out
#### Image Gen End

weightVarScale = 0.25
bias_stddev = 0.01

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data,mean = 0.0, std=bias_stddev)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def calImageGradients(images):
    # x is a 4-D tensor
    dx = images[:, :, 1:, :] - images[:, :, :-1, :]
    dy = images[:, :, :, 1:] - images[:, :, :, :-1]
    return dx, dy


perc_loss = VGGPerceptualLoss() 
perc_loss = perc_loss.to("cuda:1")

def loss_new(generated, gt):
    mse_loss = torch.nn.MSELoss()
    rmse_intensity = mse_loss(generated, gt)
    
    psnr_intensity = torch.log10(rmse_intensity)
    # print("psnr:",psnr_intensity)
    # ssim_intensity = ssim(generated, gt)
    labels_dx, labels_dy = calImageGradients(gt)
    # print("generated:",generated.shape)
    preds_dx, preds_dy = calImageGradients(generated)
    rmse_grad_x, rmse_grad_y = mse_loss(labels_dx, preds_dx), mse_loss(labels_dy, preds_dy)
    psnr_grad_x, psnr_grad_y = torch.log10(rmse_grad_x), torch.log10(rmse_grad_y)
    # print("psnr x&y:",psnr_grad_x," ",psnr_grad_y)
    p_loss = perc_loss(generated,gt)
    # print("-psnr:",-psnr_intensity,",0.5*(psnr_grad_x + psnr_grad_y):",0.5*(psnr_grad_x + psnr_grad_y),",perc_loss:",p_loss)
    total_loss = 10 + psnr_intensity + 0.5*(psnr_grad_x + psnr_grad_y) + p_loss
    # total_loss = rmse_intensity + 0.5*(rmse_grad_x + rmse_grad_y) # + p_loss
    return total_loss

def save_checkpoints(file_path, epoch_idx, model, model_solver):
    print('[INFO] Saving checkpoint to %s ...' % ( file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'model_state_dict': model.state_dict(),
        'model_solver_state_dict': model_solver.state_dict()
    }
    torch.save(checkpoint, file_path)

mode = "train"

import pickle
def save_obj(obj, name ):
    # with open('./outputF/dict/'+ name + '.pkl', 'wb') as f:
    #     pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    torch.save(obj,'./outputF/dict/'+ name + '.pkl')
def load_obj(name):
    # with open('./outputF/dict/' + name + '.pkl', 'rb') as f:
    #     return pickle.load(f)
    return torch.load('./outputF/dict/'+ name + '.pkl')

def hook_fn_back(m, i, o):
  for grad in i:
    try:
      print("Input Grad:",m,grad.shape,grad.sum())
    except AttributeError: 
      print ("None found for Gradient")
  for grad in o:  
    try:
      print("Output Grad:",m,grad.shape,grad.sum())
    except AttributeError: 
      print ("None found for Gradient")
  print("\n")

def hook_fn_for(m, i, o):
  for grad in i:
    try:
      print("Input Feats:",m,grad.shape,grad.sum())
    except AttributeError: 
      print ("None found for Gradient")
  for grad in o:  
    try:
      print("Output Feats:",m,grad.shape,grad.sum())
    except AttributeError: 
      print ("None found for Gradient")
  print("\n")

def generatePhiMaskDict(data_json, generator):
    phi_dict = {}
    mask_dict = {}
    idx_info_dict = {}
    with open(data_json, encoding='utf-8') as file:
        dataset_desc = json.loads(file.read())
        for i in range(len(dataset_desc["focaldepth"])):
            # if i == 2:
            #     break
            idx = dataset_desc["idx"][i] 
            focaldepth = dataset_desc["focaldepth"][i]
            gazeX = dataset_desc["gazeX"][i]
            gazeY = dataset_desc["gazeY"][i]
            print("focaldepth:",focaldepth," idx:",idx," gazeX:",gazeX," gazeY:",gazeY)
            phi,mask =  generator.CalculateRetinal2LayerMappings(focaldepth,torch.tensor([gazeX, gazeY]))
            phi_dict[idx]=phi
            mask_dict[idx]=mask
            idx_info_dict[idx]=[idx,focaldepth,gazeX,gazeY]
    return phi_dict,mask_dict,idx_info_dict

if __name__ == "__main__":
    ############################## generate phi and mask in pre-training
    
    # print("generating phi and mask...")
    # phi_dict,mask_dict,idx_info_dict = generatePhiMaskDict(DATA_JSON,gen)
    # save_obj(phi_dict,"phi_1204")
    # save_obj(mask_dict,"mask_1204")
    # save_obj(idx_info_dict,"idx_info_1204")
    # print("generating phi and mask end.")
    # exit(0)
    ############################# load phi and mask in pre-training
    print("loading phi and mask ...")
    phi_dict = load_obj("phi_1204")
    mask_dict = load_obj("mask_1204")
    idx_info_dict = load_obj("idx_info_1204")
    print(len(phi_dict))
    print(len(mask_dict))
    print("loading phi and mask end") 

    #train
    train_data_loader = torch.utils.data.DataLoader(dataset=lightFieldSeqDataLoader(DATA_FILE,DATA_JSON),
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    print(len(train_data_loader))

    # exit(0)


    ################################################ val #########################################################
    # val_data_loader = torch.utils.data.DataLoader(dataset=lightFieldValDataLoader(DATA_FILE,DATA_VAL_JSON),
    #                                                 batch_size=1,
    #                                                 num_workers=0,
    #                                                 pin_memory=True,
    #                                                 shuffle=False,
    #                                                 drop_last=False)

    # print(len(val_data_loader))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # lf_model = baseline.model()
    # if torch.cuda.is_available():
    #     lf_model = torch.nn.DataParallel(lf_model).cuda()

    # checkpoint = torch.load(os.path.join(OUTPUT_DIR,"gaze-ckpt-epoch-0201.pth"))
    # lf_model.load_state_dict(checkpoint["model_state_dict"])
    # lf_model.eval()

    # print("Eval::")
    # for sample_idx, (image_set, df, gazeX, gazeY, sample_idx) in enumerate(val_data_loader):
    #     print("sample_idx::",sample_idx)
    #     with torch.no_grad():
            
    #         #reshape for input
    #         image_set = image_set.permute(0,1,4,2,3) # N LF C H W
    #         image_set = image_set.reshape(image_set.shape[0],-1,image_set.shape[3],image_set.shape[4]) # N, LFxC, H, W
    #         image_set = var_or_cuda(image_set)

    #         # print("Epoch:",epoch,",Iter:",batch_idx,",Input shape:",image_set.shape, ",Input gt:",gt.shape)
    #         output = lf_model(image_set,df,gazeX,gazeY)
    #         output1,mask = GenRetinalGazeFromLayersBatch(output, gen, sample_idx, phi_dict, mask_dict)

    #         for i in range(0, 2):
    #             output1[:,i*3:i*3+3].mul_(mask[:,i:i+1])
    #             output1[:,i*3:i*3+3].clamp_(0., 1.)
            
    #         print("output:",output.shape," df:",df[0].data, ",gazeX:",gazeX[0].data,",gazeY:", gazeY[0].data)
    #         for i in range(output1.size()[0]):
    #             save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_fac1_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
    #             save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_fac2_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
    #             save_image(output1[i][0:3].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_out1_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
    #             save_image(output1[i][3:6].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_out2_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))

    #         # save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fovea_interp_l1_%.3f.png"%(df[0].data)))
    #         # save_image(output[0][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fovea_interp_l2_%.3f.png"%(df[0].data)))
    #         # output = GenRetinalFromLayersBatch(output,conf,df,v,u)
    #         # save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"1113_interp_o%.3f.png"%(df[0].data)))
    # exit()

    ################################################ train #########################################################
    lf_model = model(FIRSST_LAYER_CHANNELS,LAST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,KERNEL_SIZE_RB,INTERLEAVE_RATE)
    lf_model.apply(weight_init_normal)

    epoch_begin = 0

    ################################ load model file
    # WEIGHTS = os.path.join(OUTPUT_DIR, 'gaze-ckpt-epoch-%04d.pth' % (101))
    # print('[INFO] Recovering from %s ...' % (WEIGHTS))
    # checkpoint = torch.load(WEIGHTS)
    # init_epoch = checkpoint['epoch_idx']
    # lf_model.load_state_dict(checkpoint['model_state_dict'])
    # epoch_begin = init_epoch + 1
    # print(lf_model)
    ############################################################

    if torch.cuda.is_available():
        # lf_model = torch.nn.DataParallel(lf_model).cuda()
        lf_model = lf_model.to('cuda:1')
    lf_model.train()
    optimizer = torch.optim.Adam(lf_model.parameters(),lr=1e-2,betas=(0.9,0.999))
    l1loss = torch.nn.L1Loss()
    # lf_model.output_layer.register_backward_hook(hook_fn_back)
    print("begin training....")
    for epoch in range(epoch_begin, NUM_EPOCH):
        for batch_idx, (image_set, gt, gt2, df, gazeX, gazeY, sample_idx) in enumerate(train_data_loader):
            # print(sample_idx.shape,df.shape,gazeX.shape,gazeY.shape) # torch.Size([2, 5])
            # print(image_set.shape,gt.shape,gt2.shape) #torch.Size([2, 5, 9, 320, 320, 3]) torch.Size([2, 5, 160, 160, 3]) torch.Size([2, 5, 160, 160, 3])
            # print(delta.shape) # delta: torch.Size([2, 4, 160, 160, 3])
            
            #reshape for input
            image_set = image_set.permute(0,1,2,5,3,4) # N S LF C H W
            image_set = image_set.reshape(image_set.shape[0],image_set.shape[1],-1,image_set.shape[4],image_set.shape[5]) # N, LFxC, H, W
            image_set = var_or_cuda(image_set)
            gt = gt.permute(0,1,4,2,3) # N S C H W
            gt = var_or_cuda(gt)

            gt2 = gt2.permute(0,1,4,2,3)
            gt2 = var_or_cuda(gt2)

            gen1 = torch.empty(gt.shape)
            gen1 = var_or_cuda(gen1)

            gen2 = torch.empty(gt2.shape)
            gen2 = var_or_cuda(gen2)

            warped = torch.empty(gt2.shape[0],gt2.shape[1]-1,gt2.shape[2],gt2.shape[3],gt2.shape[4])
            warped = var_or_cuda(warped)

            delta = torch.empty(gt2.shape[0],gt2.shape[1]-1,gt2.shape[2],gt2.shape[3],gt2.shape[4])
            delta = var_or_cuda(delta)
            
            for k in range(image_set.shape[1]):
                if k == 0:
                    lf_model.reset_hidden(image_set[:,k])
                
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                output = lf_model(image_set[:,k],df[:,k],gazeX[:,k],gazeY[:,k])
                # end.record()
                # torch.cuda.synchronize()
                # print("Model Forward:",start.elapsed_time(end))
                # print("output:",output.shape) # [2, 6, 320, 320]
                # exit()
                ########################### Use Pregen Phi and Mask ###################
                # start.record()
                output1,mask = GenRetinalGazeFromLayersBatch(output, gen, sample_idx[:,k], phi_dict, mask_dict)
                # end.record()
                # torch.cuda.synchronize()
                # print("Merge:",start.elapsed_time(end))

                # print("output1 shape:",output1.shape, "mask shape:",mask.shape)
                # output1 shape: torch.Size([2, 6, 160, 160]) mask shape: torch.Size([2, 2, 160, 160])
                for i in range(0, 2):
                    output1[:,i*3:i*3+3].mul_(mask[:,i:i+1])
                    if i == 0:
                        gt[:,k].mul_(mask[:,i:i+1])
                    if i == 1:
                        gt2[:,k].mul_(mask[:,i:i+1])
                
                gen1[:,k] = output1[:,0:3]
                gen2[:,k] = output1[:,3:6]
                if ((epoch%5== 0) or epoch == 2):
                    for i in range(output.shape[0]):
                        save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fac1_o_%.3f_%.3f_%.3f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data)))
                        save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fac2_o_%.3f_%.3f_%.3f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data)))

            ########################### Update ###################
            for i in range(1,image_set.shape[1]):
                delta[:,i-1] = gt2[:,i] - gt2[:,i]
                warped[:,i-1] = gen2[:,i]-gen2[:,i-1]
            
            optimizer.zero_grad()

            # # N S C H W
            gen1 = gen1.reshape(-1,gen1.shape[2],gen1.shape[3],gen1.shape[4])
            gen2 = gen2.reshape(-1,gen2.shape[2],gen2.shape[3],gen2.shape[4])
            gt = gt.reshape(-1,gt.shape[2],gt.shape[3],gt.shape[4])
            gt2 = gt2.reshape(-1,gt2.shape[2],gt2.shape[3],gt2.shape[4])
            warped = warped.reshape(-1,warped.shape[2],warped.shape[3],warped.shape[4])
            delta = delta.reshape(-1,delta.shape[2],delta.shape[3],delta.shape[4])


            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            loss1 = loss_new(gen1,gt)
            loss2 = loss_new(gen2,gt2)
            loss3 = l1loss(warped,delta)
            loss = loss1+loss2+loss3
            # end.record()
            # torch.cuda.synchronize()
            # print("loss comp:",start.elapsed_time(end))

            
            # start.record()
            loss.backward()
            # end.record()
            # torch.cuda.synchronize()
            # print("backward:",start.elapsed_time(end))

            # start.record()
            optimizer.step()
            # end.record()
            # torch.cuda.synchronize()
            # print("optimizer step:",start.elapsed_time(end))
            
            ## Update Prev
            print("Epoch:",epoch,",Iter:",batch_idx,",loss:",loss)
            ########################### Save #####################
            if ((epoch%5== 0) or epoch == 2): # torch.Size([2, 5, 160, 160, 3])
                for i in range(gt.size()[0]):
                    # df 2,5 
                    save_image(gen1[i].data,os.path.join(OUTPUT_DIR,"gaze_out1_o_%.3f_%.3f_%.3f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data)))
                    save_image(gen2[i].data,os.path.join(OUTPUT_DIR,"gaze_out2_o_%.3f_%.3f_%.3f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data)))
                    save_image(gt[i].data,os.path.join(OUTPUT_DIR,"gaze_test1_gt0_%.3f_%.3f_%.3f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data)))
                    save_image(gt2[i].data,os.path.join(OUTPUT_DIR,"gaze_test1_gt1_%.3f_%.3f_%.3f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data)))
            if ((epoch%100 == 0) and epoch != 0 and batch_idx==len(train_data_loader)-1):
                save_checkpoints(os.path.join(OUTPUT_DIR, 'gaze-ckpt-epoch-%04d.pth' % (epoch + 1)),epoch,lf_model,optimizer)

            ########################## test Phi and Mask ##########################
            # phi,mask =  gen.CalculateRetinal2LayerMappings(df[0],torch.tensor([gazeX[0], gazeY[0]]))
            # # print("gaze Online:",gazeX[0]," ,",gazeY[0])
            # # print("df Online:",df[0])
            # # print("idx:",int(sample_idx[0].data))
            # phi_t = phi_dict[int(sample_idx[0].data)]
            # mask_t = mask_dict[int(sample_idx[0].data)]
            # # print("idx info:",idx_info_dict[int(sample_idx[0].data)])
            # # print("phi online:", phi.shape, " phi_t:", phi_t.shape)
            # # print("mask online:", mask.shape, " mask_t:", mask_t.shape)
            # print("phi delta:", (phi-phi_t).sum()," mask delta:",(mask -mask_t).sum())
            # exit(0)

            ###########################Gen Batch 1 by 1###################
            # phi,mask =  gen.CalculateRetinal2LayerMappings(df[0],torch.tensor([gazeX[0], gazeY[0]]))
            # # print(phi.shape) # 2,240,320,41,2
            # output1, mask = GenRetinalFromLayersBatch_Online(output, gen, phi, mask)
            ###########################Gen Batch 1 by 1###################