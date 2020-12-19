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
from loss import *
import json

from baseline import *
from data import * 

import torch.autograd.profiler as profiler
# param
BATCH_SIZE = 2
NUM_EPOCH = 1001
INTERLEAVE_RATE = 2
IM_H = 540
IM_W = 376
Retinal_IM_H = 540
Retinal_IM_W = 376
N = 4 # number of input light field stack
M = 1 # number of display layers
DATA_FILE = "/home/yejiannan/Project/deeplightfield/data/lf_syn"
DATA_JSON = "/home/yejiannan/Project/deeplightfield/data/data_lf_syn_full.json"
# DATA_VAL_JSON = "/home/yejiannan/Project/LightField/data/data_gaze_fovea_val.json"
OUTPUT_DIR = "/home/yejiannan/Project/deeplightfield/outputE/lf_syn_full1219"
OUT_CHANNELS_RB = 128
KERNEL_SIZE_RB = 3
KERNEL_SIZE = 3
LAST_LAYER_CHANNELS = 3 * INTERLEAVE_RATE**2
FIRSST_LAYER_CHANNELS = 12 * INTERLEAVE_RATE**2

from weight_init import weight_init_normal

def save_checkpoints(file_path, epoch_idx, model, model_solver):
    print('[INFO] Saving checkpoint to %s ...' % ( file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'model_state_dict': model.state_dict(),
        'model_solver_state_dict': model_solver.state_dict()
    }
    torch.save(checkpoint, file_path)

mode = "Silence" #"Perf"
w_frame = 1.0
loss1 = PerceptionReconstructionLoss()
if __name__ == "__main__":
    #train
    train_data_loader = torch.utils.data.DataLoader(dataset=lightFieldSynDataLoader(DATA_FILE,DATA_JSON),
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    #Data loader test
    print(len(train_data_loader))

    lf_model = model(FIRSST_LAYER_CHANNELS,LAST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,KERNEL_SIZE_RB,INTERLEAVE_RATE,RNN=False)
    lf_model.apply(weight_init_normal)
    lf_model.train()
    epoch_begin = 0

    if torch.cuda.is_available():
        # lf_model = torch.nn.DataParallel(lf_model).cuda()
        lf_model = lf_model.to('cuda:1')
    
    optimizer = torch.optim.Adam(lf_model.parameters(),lr=5e-3,betas=(0.9,0.999))
    
    # lf_model.output_layer.register_backward_hook(hook_fn_back)
    if mode=="Perf":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    print("begin training....")
    for epoch in range(epoch_begin, NUM_EPOCH):
        for batch_idx, (image_set, gt, pos_row, pos_col) in enumerate(train_data_loader):
            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("load:",start.elapsed_time(end))

                start.record()
            #reshape for input
            image_set = image_set.permute(0,1,4,2,3) # N LF C H W 
            image_set = image_set.reshape(image_set.shape[0],-1,image_set.shape[3],image_set.shape[4]) # N LFxC H W
            image_set = var_or_cuda(image_set)

            gt = gt.permute(0,3,1,2) # BS C H W 
            gt = var_or_cuda(gt)

            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("data prepare:",start.elapsed_time(end))

                start.record()
            
            output = lf_model(image_set,pos_row, pos_col) # 2 6 376 540

            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("forward:",start.elapsed_time(end))

                start.record()
            optimizer.zero_grad()
            # print("output:",output.shape," gt:",gt.shape)
            loss1_value = loss1(output,gt)
            loss = (w_frame * loss1_value)

            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("compute loss:",start.elapsed_time(end))

                start.record()
            loss.backward()
            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("backward:",start.elapsed_time(end))

                start.record()
            optimizer.step()
            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("update:",start.elapsed_time(end))

            print("Epoch:",epoch,",Iter:",batch_idx,",loss:",loss.item())
            
            # exit(0)
            ########################### Save #####################
            if ((epoch%10== 0 and epoch != 0) or epoch == 2): # torch.Size([2, 5, 160, 160, 3])
                for i in range(gt.size()[0]):
                    save_image(output[i].data,os.path.join(OUTPUT_DIR,"out_%.5f_%.5f.png"%(pos_col[i].data,pos_row[i].data)))
                    save_image(gt[i].data,os.path.join(OUTPUT_DIR,"gt_%.5f_%.5f.png"%(pos_col[i].data,pos_row[i].data)))
            if ((epoch%100 == 0) and epoch != 0 and batch_idx==len(train_data_loader)-1):
                save_checkpoints(os.path.join(OUTPUT_DIR, 'ckpt-epoch-%04d.pth' % (epoch)),epoch,lf_model,optimizer)