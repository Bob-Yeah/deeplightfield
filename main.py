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
# param
BATCH_SIZE = 16
NUM_EPOCH = 1000

INTERLEAVE_RATE = 2

IM_H = 320
IM_W = 320

Retinal_IM_H = 320
Retinal_IM_W = 320

N = 9 # number of input light field stack
M = 2 # number of display layers

DATA_FILE = "/home/yejiannan/Project/LightField/data/gaze_small_nar_new"
DATA_JSON = "/home/yejiannan/Project/LightField/data/data_gaze_low_new.json"
DATA_VAL_JSON = "/home/yejiannan/Project/LightField/data/data_val.json"
OUTPUT_DIR = "/home/yejiannan/Project/LightField/output/gaze_low_new_1125_minibatch"

class lightFieldDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path, file_json, transforms=None):
        self.file_dir_path = file_dir_path
        self.transforms = transforms
        with open(file_json, encoding='utf-8') as file:
            self.dataset_desc = json.loads(file.read())

    def __len__(self):
        return len(self.dataset_desc["focaldepth"])

    def __getitem__(self, idx):
        lightfield_images, gt, fd, gazeX, gazeY, sample_idx = self.get_datum(idx)
        if self.transforms:
            lightfield_images = self.transforms(lightfield_images)
        return (lightfield_images, gt, fd, gazeX, gazeY, sample_idx)

    def get_datum(self, idx):
        lf_image_paths = os.path.join(DATA_FILE, self.dataset_desc["train"][idx])
        fd_gt_path = os.path.join(DATA_FILE, self.dataset_desc["gt"][idx])
        lf_images = []
        lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)
        for i in range(9):
            lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:3]
            ## IF GrayScale
            # lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:1]
            # print(lf_image.shape)
            lf_images.append(lf_image)
        gt = cv2.imread(fd_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        ## IF GrayScale
        # gt = cv2.imread(fd_gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        # gt = np.expand_dims(gt, axis=-1)

        fd = self.dataset_desc["focaldepth"][idx]
        gazeX = self.dataset_desc["gazeX"][idx]
        gazeY = self.dataset_desc["gazeY"][idx]
        sample_idx = self.dataset_desc["idx"][idx]
        return np.asarray(lf_images),gt,fd,gazeX,gazeY,sample_idx

OUT_CHANNELS_RB = 128
KERNEL_SIZE_RB = 3
KERNEL_SIZE = 3

class residual_block(torch.nn.Module):
    def __init__(self,delta_channel_dim):
        super(residual_block,self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(OUT_CHANNELS_RB+delta_channel_dim,OUT_CHANNELS_RB+delta_channel_dim,KERNEL_SIZE_RB,stride=1,padding = 1),
            torch.nn.BatchNorm2d(OUT_CHANNELS_RB+delta_channel_dim),
            torch.nn.ELU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(OUT_CHANNELS_RB+delta_channel_dim,OUT_CHANNELS_RB+delta_channel_dim,KERNEL_SIZE_RB,stride=1,padding = 1),
            torch.nn.BatchNorm2d(OUT_CHANNELS_RB+delta_channel_dim),
            torch.nn.ELU()
        )

    def forward(self,input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = input+output
        return output

class deinterleave(torch.nn.Module):
    def __init__(self, block_size):
        super(deinterleave, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class interleave(torch.nn.Module):
    def __init__(self, block_size):
        super(interleave, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

LAST_LAYER_CHANNELS = 6 * INTERLEAVE_RATE**2
FIRSST_LAYER_CHANNELS = 27 * INTERLEAVE_RATE**2

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.interleave = interleave(INTERLEAVE_RATE)

        self.first_layer = torch.nn.Sequential(
            torch.nn.Conv2d(FIRSST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,stride=1,padding=1),
            torch.nn.BatchNorm2d(OUT_CHANNELS_RB),
            torch.nn.ELU()
        )
        
        self.residual_block1 = residual_block(0)
        self.residual_block2 = residual_block(3)
        self.residual_block3 = residual_block(3)
        self.residual_block4 = residual_block(3)
        self.residual_block5 = residual_block(3)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(OUT_CHANNELS_RB+3,LAST_LAYER_CHANNELS,KERNEL_SIZE,stride=1,padding=1),
            torch.nn.BatchNorm2d(LAST_LAYER_CHANNELS),
            torch.nn.Sigmoid()
        )
        self.deinterleave = deinterleave(INTERLEAVE_RATE)


    def forward(self, lightfield_images, focal_length, gazeX, gazeY):
        input_to_net = self.interleave(lightfield_images)
        input_to_rb = self.first_layer(input_to_net)
        output = self.residual_block1(input_to_rb)
        depth_layer = torch.ones((input_to_rb.shape[0],1,input_to_rb.shape[2],input_to_rb.shape[3]))
        gazeX_layer = torch.ones((input_to_rb.shape[0],1,input_to_rb.shape[2],input_to_rb.shape[3]))
        gazeY_layer = torch.ones((input_to_rb.shape[0],1,input_to_rb.shape[2],input_to_rb.shape[3]))
        for i in range(focal_length.shape[0]):
            depth_layer[i] *= 1. / focal_length[i]
            gazeX_layer[i] *= (gazeX[i] - (-3.333)) / (3.333*2)
            gazeY_layer[i] *= (gazeY[i] - (-3.333)) / (3.333*2)
        depth_layer = var_or_cuda(depth_layer)
        gazeX_layer = var_or_cuda(gazeX_layer)
        gazeY_layer = var_or_cuda(gazeY_layer)

        output = torch.cat((output,depth_layer,gazeX_layer,gazeY_layer),dim=1)
        output = self.residual_block2(output)
        output = self.residual_block3(output)
        output = self.residual_block4(output)
        output = self.residual_block5(output)
        output = self.output_layer(output)
        output = self.deinterleave(output)
        return output

class Conf(object):
    def __init__(self):
        self.pupil_size = 0.02 # 2cm
        self.retinal_res = torch.tensor([ Retinal_IM_H, Retinal_IM_W ])
        self.layer_res = torch.tensor([ IM_H, IM_W ])
        self.layer_hfov = 90    # layers' horizontal FOV
        self.eye_hfov = 85      # eye's horizontal FOV
        self.d_layer = [ 1, 3 ] # layers' distance
        
    def GetNLayers(self):
        return len(self.d_layer)
    
    def GetLayerSize(self, i):
        w = Fov2Length(self.layer_hfov)
        h = w * self.layer_res[0] / self.layer_res[1]
        return torch.tensor([ h, w ]) * self.d_layer[i]

    def GetEyeViewportSize(self): 
        w = Fov2Length(self.eye_hfov) 
        h = w * self.retinal_res[0] / self.retinal_res[1] 
        return torch.tensor([ h, w ])

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
        phi = var_or_cuda(phi)
        phi.requires_grad = False
        retinal[i] = gen.GenRetinalFromLayers(layers[i],phi)
        mask.append(mask_dict[int(sample_idx[i].data)])
    retinal = var_or_cuda(retinal)
    mask = torch.stack(mask,dim = 0).unsqueeze(1) # batch x 1 x height x width
    return retinal, mask

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

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def calImageGradients(images):
    # x is a 4-D tensor
    dx = images[:, :, 1:, :] - images[:, :, :-1, :]
    dy = images[:, :, :, 1:] - images[:, :, :, :-1]
    return dx, dy


perc_loss = VGGPerceptualLoss() 
perc_loss = perc_loss.to("cuda")

def loss_new(generated, gt):
    mse_loss = torch.nn.MSELoss()
    rmse_intensity = mse_loss(generated, gt)
    
    psnr_intensity = torch.log10(rmse_intensity)
    labels_dx, labels_dy = calImageGradients(gt)
    preds_dx, preds_dy = calImageGradients(generated)
    rmse_grad_x, rmse_grad_y = mse_loss(labels_dx, preds_dx), mse_loss(labels_dy, preds_dy)
    psnr_grad_x, psnr_grad_y = torch.log10(rmse_grad_x), torch.log10(rmse_grad_y)
    p_loss = perc_loss(generated,gt)
    total_loss = 10 + psnr_intensity + 0.5*(psnr_grad_x + psnr_grad_y) + p_loss
    return total_loss

def save_checkpoints(file_path, epoch_idx, model, model_solver):
    print('[INFO] Saving checkpoint to %s ...' % ( file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'model_state_dict': model.state_dict(),
        'model_solver_state_dict': model_solver.state_dict()
    }
    torch.save(checkpoint, file_path)

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

if __name__ == "__main__":

    ############################## generate phi and mask in pre-training
    phi_dict = {}
    mask_dict = {}
    idx_info_dict = {}
    print("generating phi and mask...")
    with open(DATA_JSON, encoding='utf-8') as file:
        dataset_desc = json.loads(file.read())
        for i in range(len(dataset_desc["focaldepth"])):
            # if i == 2:
            #     break
            idx = dataset_desc["idx"][i] 
            focaldepth = dataset_desc["focaldepth"][i]
            gazeX = dataset_desc["gazeX"][i]
            gazeY = dataset_desc["gazeY"][i]
            # print("focaldepth:",focaldepth," idx:",idx," gazeX:",gazeX," gazeY:",gazeY)
            phi,mask =  gen.CalculateRetinal2LayerMappings(focaldepth,torch.tensor([gazeX, gazeY]))
            phi_dict[idx]=phi
            mask_dict[idx]=mask
            idx_info_dict[idx]=[idx,focaldepth,gazeX,gazeY]
    print("generating phi and mask end.")
    
    # exit(0)
    #train
    train_data_loader = torch.utils.data.DataLoader(dataset=lightFieldDataLoader(DATA_FILE,DATA_JSON),
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    print(len(train_data_loader))

    # exit(0)

    ################################################ train #########################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lf_model = model()
    lf_model.apply(weight_init_normal)

    epoch_begin = 0

    if torch.cuda.is_available():
        lf_model = torch.nn.DataParallel(lf_model).cuda()
    lf_model.train()
    optimizer = torch.optim.Adam(lf_model.parameters(),lr=1e-2,betas=(0.9,0.999))

    print("begin training....")
    for epoch in range(epoch_begin, NUM_EPOCH):
        for batch_idx, (image_set, gt, df, gazeX, gazeY, sample_idx) in enumerate(train_data_loader):
            #reshape for input
            image_set = image_set.permute(0,1,4,2,3) # N LF C H W
            image_set = image_set.reshape(image_set.shape[0],-1,image_set.shape[3],image_set.shape[4]) # N, LFxC, H, W
            image_set = var_or_cuda(image_set)
            gt = gt.permute(0,3,1,2)
            gt = var_or_cuda(gt)

            optimizer.zero_grad()
            output = lf_model(image_set,df,gazeX,gazeY)
            ########################### Use Pregen Phi and Mask ###################
            output1,mask = GenRetinalFromLayersBatch(output, gen, sample_idx, phi_dict, mask_dict)
            mask = var_or_cuda(mask)
            mask.requires_grad = False
            output_f = output1 * mask
            gt = gt * mask
            loss = loss_new(output_f,gt)
            print("Epoch:",epoch,",Iter:",batch_idx,",loss:",loss)

            ########################### Update ###################
            loss.backward()
            optimizer.step()

            ########################### Save #####################
            if ((epoch%50== 0) or epoch == 5):
                for i in range(output_f.size()[0]):
                    save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fac1_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
                    save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fac2_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
                    save_image(output_f[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_test1_o_%.3f_%.3f_%.3f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
            if ((epoch%200 == 0) and epoch != 0 and batch_idx==len(train_data_loader)-1):
                save_checkpoints(os.path.join(OUTPUT_DIR, 'gaze-ckpt-epoch-%04d.pth' % (epoch + 1)),
                                epoch,lf_model,optimizer)