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
BATCH_SIZE = 5
NUM_EPOCH = 5000

INTERLEAVE_RATE = 2

IM_H = 480
IM_W = 640

N = 9 # number of input light field stack
M = 2 # number of display layers

DATA_FILE = "/home/yejiannan/Project/LightField/data/try"
DATA_JSON = "/home/yejiannan/Project/LightField/data/data.json"
DATA_VAL_JSON = "/home/yejiannan/Project/LightField/data/data_val.json"
OUTPUT_DIR = "/home/yejiannan/Project/LightField/output"

class lightFieldDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path, file_json, transforms=None):
        self.file_dir_path = file_dir_path
        self.transforms = transforms
        # self.datum_list = glob.glob(os.path.join(file_dir_path,"*"))
        with open(file_json, encoding='utf-8') as file:
            self.dastset_desc = json.loads(file.read())

    def __len__(self):
        return len(self.dastset_desc["focaldepth"])

    def __getitem__(self, idx):
        lightfield_images, gt, fd = self.get_datum(idx)
        if self.transforms:
            lightfield_images = self.transforms(lightfield_images)
        return (lightfield_images, gt, fd)

    def get_datum(self, idx):
        lf_image_paths = os.path.join(DATA_FILE, self.dastset_desc["train"][idx])
        # print(lf_image_paths)
        fd_gt_path = os.path.join(DATA_FILE, self.dastset_desc["gt"][idx])
        # print(fd_gt_path)
        lf_images = []
        lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)
        for i in range(9):
            lf_image = lf_image_big[i//3*IM_H:i//3*IM_H+IM_H,i%3*IM_W:i%3*IM_W+IM_W,0:3]
            # print(lf_image.shape)
            lf_images.append(lf_image)
        gt = cv2.imread(fd_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        fd = self.dastset_desc["focaldepth"][idx]
        return (np.asarray(lf_images),gt,fd)

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
        self.residual_block2 = residual_block(1)
        self.residual_block3 = residual_block(1)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(OUT_CHANNELS_RB+1,LAST_LAYER_CHANNELS,KERNEL_SIZE,stride=1,padding=1),
            torch.nn.BatchNorm2d(LAST_LAYER_CHANNELS),
            torch.nn.Sigmoid()
        )
        self.deinterleave = deinterleave(INTERLEAVE_RATE)


    def forward(self, lightfield_images, focal_length):
        # lightfield_images: torch.Size([batch_size, channels * D, H, W]) 
        # channels : RGB*D: 3*9, H:256, W:256
        input_to_net = self.interleave(lightfield_images)
        # print("after interleave:",input_to_net.shape)
        input_to_rb = self.first_layer(input_to_net)
        output = self.residual_block1(input_to_rb)
        # print("output1:",output.shape)
        
        depth_layer = torch.ones((output.shape[0],1,output.shape[2],output.shape[3]))
        # print(df.shape[0])
        for i in range(focal_length.shape[0]):
            depth_layer[i] = 1. / focal_length[i]
            # print(depth_layer.shape)
        depth_layer = var_or_cuda(depth_layer)
        output = torch.cat((output,depth_layer),dim=1)

        output = self.residual_block2(output)
        output = self.residual_block3(output)
        # output = output + input_to_net
        output = self.output_layer(output)
        output = self.deinterleave(output)
        return output

class Conf(object):
    def __init__(self):
        self.pupil_size = 0.02 # 2cm
        self.retinal_res = torch.tensor([ 480, 640 ])
        self.layer_res = torch.tensor([ 480, 640 ])
        self.n_layers = 2
        self.d_layer = [ 1., 3. ] # layers' distance
        self.h_layer = [ 1. * 480. / 640., 3. * 480. / 640. ] # layers' height

#### Image Gen
conf = Conf()

v = torch.tensor([conf.h_layer[0] / conf.d_layer[0],
     conf.h_layer[0] / conf.d_layer[0] * conf.layer_res[1] / conf.layer_res[0]])

u = GenSamplesInPupil(conf, 5)

def GenRetinalFromLayersBatch(layers, conf, df, v, u):
    # layers: batchsize, 2, color, height, width 
    # Phi:torch.Size([batchsize, 480, 640, 2, 41, 2])
    # df : batchsize,..
    H_r = conf.retinal_res[0]
    W_r = conf.retinal_res[1]
    D_r = conf.retinal_res.double()
    N = conf.n_layers
    M = u.size()[0] #41
    BS = df.shape[0]
    Phi = torch.empty(BS, H_r, W_r, N, M, 2, dtype=torch.long)
    # print("Phi:",Phi.shape)

    p_rx, p_ry = torch.meshgrid(torch.tensor(range(0, H_r)),
                                torch.tensor(range(0, W_r)))
    p_r = torch.stack([p_rx, p_ry], 2).unsqueeze(2).expand(-1, -1, M, -1)
    # print("p_r:",p_r.shape) #torch.Size([480, 640, 41, 2])
    for bs in range(BS):
        for i in range(0, N):
            dpi = conf.h_layer[i] / float(conf.layer_res[0]) # 1 / 480
            # print("dpi:",dpi)
            ci = conf.layer_res / 2 # [240,320]
            di = conf.d_layer[i] # 深度
            pi_r = di * v * (1. / D_r * (p_r + 0.5) - 0.5) / dpi # [480, 640, 41, 2]
            wi = (1 - di / df[bs]) / dpi # (1 - 深度/聚焦) / dpi  df = 2.625 di = 1.75
            pi = torch.floor(pi_r + ci + wi * u)
            torch.clamp_(pi[:, :, :, 0], 0, conf.layer_res[0] - 1)
            torch.clamp_(pi[:, :, :, 1], 0, conf.layer_res[1] - 1)
            Phi[bs, :, :, i, :, :] = pi
    # print("Phi slice:",Phi[0, :, :, 0, 0, 0].shape)
    retinal = torch.ones(BS, 3, H_r, W_r)
    retinal = var_or_cuda(retinal)
    for bs in range(BS):
        for j in range(0, M):
            retinal_view = torch.ones(3, H_r, W_r)
            retinal_view = var_or_cuda(retinal_view)
            for i in range(0, N):
                retinal_view.mul_(layers[bs, (i * 3) : (i * 3 + 3), Phi[bs, :, :, i, j, 0], Phi[bs, :, :, i, j, 1]])
            retinal[bs,:,:,:].add_(retinal_view)
        retinal[bs,:,:,:].div_(M)
    return retinal
#### Image Gen End

def merge_two(near,far):
    df = conf.d_layer[0] + (conf.d_layer[1] - conf.d_layer[0]) / 2.
    # Phi = GenRetinal2LayerMappings(conf, df, v, u)
    # retinal = GenRetinalFromLayers(layers, Phi)
    return near[:,0:3,:,:] + far[:,3:6,:,:] / 2.0

def loss_two_images(generated, gt):
    l1_loss = torch.nn.L1Loss()
    return l1_loss(generated, gt)

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
    dy = images[:, 1:, :, :] - images[:, :-1, :, :]
    return dx, dy


perc_loss = VGGPerceptualLoss() 
perc_loss = perc_loss.to("cuda")

def loss_new(generated, gt):
    mse_loss = torch.nn.MSELoss()
    rmse_intensity = mse_loss(generated, gt)
    RENORM_SCALE = torch.tensor(0.9)
    RENORM_SCALE = var_or_cuda(RENORM_SCALE)
    psnr_intensity = torch.log10(rmse_intensity)
    ssim_intensity = ssim(generated, gt)
    labels_dx, labels_dy = calImageGradients(gt)
    preds_dx, preds_dy = calImageGradients(generated)
    rmse_grad_x, rmse_grad_y = mse_loss(labels_dx, preds_dx), mse_loss(labels_dy, preds_dy)
    psnr_grad_x, psnr_grad_y = torch.log10(rmse_grad_x), torch.log10(rmse_grad_y)
    p_loss = perc_loss(generated,gt)
    # print("-psnr:",-psnr_intensity,",0.5*(psnr_grad_x + psnr_grad_y):",0.5*(psnr_grad_x + psnr_grad_y),",perc_loss:",p_loss)
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

mode = "val"
if __name__ == "__main__":
    #test
    # train_dataset = lightFieldDataLoader(DATA_FILE,DATA_JSON)
    # print(train_dataset[0][0].shape)
    # cv2.imwrite("test_crop0.png",train_dataset[0][1]*255.)
    # save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"o%d_%d.png"%(epoch,batch_idx)))
    #test end
    
    #train
    train_data_loader = torch.utils.data.DataLoader(dataset=lightFieldDataLoader(DATA_FILE,DATA_JSON),
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    print(len(train_data_loader))

    val_data_loader = torch.utils.data.DataLoader(dataset=lightFieldDataLoader(DATA_FILE,DATA_VAL_JSON),
                                                    batch_size=1,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    shuffle=False,
                                                    drop_last=False)

    print(len(val_data_loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lf_model = model()
    if torch.cuda.is_available():
        lf_model = torch.nn.DataParallel(lf_model).cuda()

    #val 
    checkpoint = torch.load(os.path.join(OUTPUT_DIR,"ckpt-epoch-3001.pth"))
    lf_model.load_state_dict(checkpoint["model_state_dict"])
    lf_model.eval()

    print("Eval::")
    for sample_idx, (image_set, gt, df) in enumerate(val_data_loader):
        print("sample_idx::")
        with torch.no_grad():
            #reshape for input
            image_set = image_set.permute(0,1,4,2,3) # N LF C H W
            image_set = image_set.reshape(image_set.shape[0],-1,image_set.shape[3],image_set.shape[4]) # N, LFxC, H, W
            image_set = var_or_cuda(image_set)
            # image_set.to(device)
            gt = gt.permute(0,3,1,2)
            gt = var_or_cuda(gt)
            # print("Epoch:",epoch,",Iter:",batch_idx,",Input shape:",image_set.shape, ",Input gt:",gt.shape)
            output = lf_model(image_set,df)
            print("output:",output.shape," df:",df)
            save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"1113_interp_l1_%.3f.png"%(df[0].data)))
            save_image(output[0][3:6].data,os.path.join(OUTPUT_DIR,"1113_interp_l2_%.3f.png"%(df[0].data)))
            output = GenRetinalFromLayersBatch(output,conf,df,v,u)
            save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"1113_interp_o%.3f.png"%(df[0].data)))
    exit()

    # train
    # print(lf_model)
    # exit()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # lf_model = model()
    # lf_model.apply(weight_init_normal)

    # if torch.cuda.is_available():
    #     lf_model = torch.nn.DataParallel(lf_model).cuda()
    # lf_model.train()
    # optimizer = torch.optim.Adam(lf_model.parameters(),lr=5e-2,betas=(0.9,0.999))
    
    # for epoch in range(NUM_EPOCH):
    #     for batch_idx, (image_set, gt, df) in enumerate(train_data_loader):
    #         #reshape for input
    #         image_set = image_set.permute(0,1,4,2,3) # N LF C H W
    #         image_set = image_set.reshape(image_set.shape[0],-1,image_set.shape[3],image_set.shape[4]) # N, LFxC, H, W
            
    #         image_set = var_or_cuda(image_set)
    #         # image_set.to(device)
    #         gt = gt.permute(0,3,1,2)
    #         gt = var_or_cuda(gt)
    #         # print("Epoch:",epoch,",Iter:",batch_idx,",Input shape:",image_set.shape, ",Input gt:",gt.shape)
    #         optimizer.zero_grad()
    #         output = lf_model(image_set,df)
    #         # print("output:",output.shape," df:",df.shape)
    #         output = GenRetinalFromLayersBatch(output,conf,df,v,u)
    #         loss = loss_new(output,gt)
    #         print("Epoch:",epoch,",Iter:",batch_idx,",loss:",loss)
    #         loss.backward()
    #         optimizer.step()
    #         if (epoch%100 == 0):
    #             for i in range(BATCH_SIZE):
    #                 save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"cuda_lr_5e-2_mul_dip_newloss_debug_conf_o%d_%d.png"%(epoch,i)))
    #         if (epoch%1000 == 0):
    #             save_checkpoints(os.path.join(OUTPUT_DIR, 'ckpt-epoch-%04d.pth' % (epoch + 1)),
    #                             epoch,lf_model,optimizer)
                
                