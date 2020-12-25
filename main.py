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
from .gen_image import *
from .loss import *
import json
from .conf import Conf

from .baseline import *
from .data import * 

import torch.autograd.profiler as profiler
# param
BATCH_SIZE = 1
NUM_EPOCH = 1001
INTERLEAVE_RATE = 2
IM_H = 320
IM_W = 320
Retinal_IM_H = 320
Retinal_IM_W = 320
N = 25 # number of input light field stack
M = 2 # number of display layers
DATA_FILE = "/home/yejiannan/Project/LightField/data/FlowRPG1211"
DATA_JSON = "/home/yejiannan/Project/LightField/data/data_gaze_fovea_seq_flow_RPG.json"
# DATA_VAL_JSON = "/home/yejiannan/Project/LightField/data/data_gaze_fovea_val.json"
OUTPUT_DIR = "/home/yejiannan/Project/LightField/outputE/gaze_fovea_seq_flow_RPG_seq5_same_loss"
OUT_CHANNELS_RB = 128
KERNEL_SIZE_RB = 3
KERNEL_SIZE = 3
LAST_LAYER_CHANNELS = 6 * INTERLEAVE_RATE**2
FIRSST_LAYER_CHANNELS = 75 * INTERLEAVE_RATE**2

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

import time
def GenRetinalGazeFromLayersBatchSpeed(layers, gen, phi, phi_invalid, retinal_invalid):
    # layers: batchsize, 2*color, height, width 
    # Phi:torch.Size([batchsize, Layer, h, w,  41, 2])
    # df : batchsize,..
    # start1 = time.time()
    #  retinal bs x color x height x width
    retinal_fovea = torch.empty((layers.shape[0], 6, 160, 160),device="cuda:2")
    mask_fovea = torch.empty((layers.shape[0], 2, 160, 160),device="cuda:2")
    # start = time.time()
    retinal = gen.GenRetinalFromLayersBatch(layers,phi_batch)
    # print("retinal:",retinal.shape) #retinal: torch.Size([2, 3, 320, 320])
    # print("t2:",time.time() - start)

    # start = time.time()
    fovea_layers, fovea_layer_masks = gen.GenFoveaLayersBatch(retinal,mask_batch)

    mask_fovea = torch.stack([fovea_layer_masks[0],fovea_layer_masks[1]],dim=1)
    retinal_fovea = torch.cat([fovea_layers[0],fovea_layers[1]],dim=1)
    # print("t3:",time.time() - start)

    retinal_fovea = var_or_cuda(retinal_fovea)
    mask_fovea = var_or_cuda(mask_fovea) # batch x 2 x height x width
    # mask = torch.stack(mask,dim = 0).unsqueeze(1) 
    return retinal_fovea, mask_fovea

def MergeBatchSpeed(layers, gen, phi, phi_invalid, retinal_invalid):
    # layers: batchsize, 2*color, height, width 
    # Phi:torch.Size([batchsize, Layer, h, w,  41, 2])
    # df : batchsize,..
    # start1 = time.time()
    #  retinal bs x color x height x width
    # retinal_fovea = torch.empty((layers.shape[0], 6, 160, 160),device="cuda:2")
    # mask_fovea = torch.empty((layers.shape[0], 2, 160, 160),device="cuda:2")
    # start = time.time()
    retinal = gen.GenRetinalFromLayersBatch(layers,phi) #retinal: torch.Size([BatchSize , 3, 320, 320])
    retinal.mul_(~retinal_invalid.to("cuda:2"))
    # print("retinal:",retinal.shape) 
    # print("t2:",time.time() - start)
    
    # start = time.time()
    # fovea_layers, fovea_layer_masks = gen.GenFoveaLayersBatch(retinal,mask_batch)

    # mask_fovea = torch.stack([fovea_layer_masks[0],fovea_layer_masks[1]],dim=1)
    # retinal_fovea = torch.cat([fovea_layers[0],fovea_layers[1]],dim=1)
    # print("t3:",time.time() - start)

    # retinal_fovea = var_or_cuda(retinal_fovea)
    # mask_fovea = var_or_cuda(mask_fovea) # batch x 2 x height x width
    # mask = torch.stack(mask,dim = 0).unsqueeze(1) 
    return retinal

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

from weight_init import weight_init_normal

def save_checkpoints(file_path, epoch_idx, model, model_solver):
    print('[INFO] Saving checkpoint to %s ...' % ( file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'model_state_dict': model.state_dict(),
        'model_solver_state_dict': model_solver.state_dict()
    }
    torch.save(checkpoint, file_path)

# import pickle
# def save_obj(obj, name ):
#     # with open('./outputF/dict/'+ name + '.pkl', 'wb') as f:
#     #     pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#     torch.save(obj,'./outputF/dict/'+ name + '.pkl')
# def load_obj(name):
#     # with open('./outputF/dict/' + name + '.pkl', 'rb') as f:
#     #     return pickle.load(f)
#     return torch.load('./outputF/dict/'+ name + '.pkl')

# def generatePhiMaskDict(data_json, generator):
#     phi_dict = {}
#     mask_dict = {}
#     idx_info_dict = {}
#     with open(data_json, encoding='utf-8') as file:
#         dataset_desc = json.loads(file.read())
#         for i in range(len(dataset_desc["focaldepth"])):
#             # if i == 2:
#             #     break
#             idx = dataset_desc["idx"][i] 
#             focaldepth = dataset_desc["focaldepth"][i]
#             gazeX = dataset_desc["gazeX"][i]
#             gazeY = dataset_desc["gazeY"][i]
#             print("focaldepth:",focaldepth," idx:",idx," gazeX:",gazeX," gazeY:",gazeY)
#             phi,mask =  generator.CalculateRetinal2LayerMappings(focaldepth,torch.tensor([gazeX, gazeY]))
#             phi_dict[idx]=phi
#             mask_dict[idx]=mask
#             idx_info_dict[idx]=[idx,focaldepth,gazeX,gazeY]
#     return phi_dict,mask_dict,idx_info_dict

# def generatePhiMaskDictNew(data_json, generator):
#     phi_dict = {}
#     mask_dict = {}
#     idx_info_dict = {}
#     with open(data_json, encoding='utf-8') as file:
#         dataset_desc = json.loads(file.read())
#         for i in range(len(dataset_desc["seq"])):
#             for j in dataset_desc["seq"][i]:
#                 idx = dataset_desc["idx"][j] 
#                 focaldepth = dataset_desc["focaldepth"][j]
#                 gazeX = dataset_desc["gazeX"][j]
#                 gazeY = dataset_desc["gazeY"][j]
#                 print("focaldepth:",focaldepth," idx:",idx," gazeX:",gazeX," gazeY:",gazeY)
#                 phi,mask =  generator.CalculateRetinal2LayerMappings(focaldepth,torch.tensor([gazeX, gazeY]))
#                 phi_dict[idx]=phi
#                 mask_dict[idx]=mask
#                 idx_info_dict[idx]=[idx,focaldepth,gazeX,gazeY]
#     return phi_dict,mask_dict,idx_info_dict

mode = "Silence" #"Perf"
model_type = "RNN" #"RNN"
w_frame = 0.9
w_inter_frame = 0.1
batch_model = "NoSingle"
loss1 = ReconstructionLoss()
loss2 = ReconstructionLoss()
if __name__ == "__main__":
    ############################## generate phi and mask in pre-training
    
    # print("generating phi and mask...")
    # phi_dict,mask_dict,idx_info_dict = generatePhiMaskDictNew(DATA_JSON,gen)
    # # save_obj(phi_dict,"phi_1204")
    # # save_obj(mask_dict,"mask_1204")
    # # save_obj(idx_info_dict,"idx_info_1204")
    # print("generating phi and mask end.")
    # exit(0)
    ############################# load phi and mask in pre-training
    # print("loading phi and mask ...")
    # phi_dict = load_obj("phi_1204")
    # mask_dict = load_obj("mask_1204")
    # idx_info_dict = load_obj("idx_info_1204")
    # print(len(phi_dict))
    # print(len(mask_dict))
    # print("loading phi and mask end") 

    #### Image Gen and conf 
    conf = Conf()
    gen = RetinalGen(conf)

    #train
    train_data_loader = torch.utils.data.DataLoader(dataset=lightFieldFlowSeqDataLoader(DATA_FILE,DATA_JSON, gen, conf),
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    #Data loader test
    print(len(train_data_loader))

    # # lightfield_images, gt, flow, fd, gazeX, gazeY, posX, posY, sample_idx, phi, mask
    # # lightfield_images, gt, phi, phi_invalid, retinal_invalid, flow, fd, gazeX, gazeY, posX, posY, posZ, sample_idx
    # for batch_idx, (image_set, gt,phi, phi_invalid, retinal_invalid, flow, df, gazeX, gazeY, posX, posY, posZ, sample_idx) in enumerate(train_data_loader):
    #     print(image_set.shape,type(image_set))
    #     print(gt.shape,type(gt))
    #     print(phi.shape,type(phi))
    #     print(phi_invalid.shape,type(phi_invalid))
    #     print(retinal_invalid.shape,type(retinal_invalid))
    #     print(flow.shape,type(flow))
    #     print(df.shape,type(df))
    #     print(gazeX.shape,type(gazeX))
    #     print(posX.shape,type(posX))
    #     print(sample_idx.shape,type(sample_idx))
    #     print("test train dataloader.")
    #     exit(0)
    #Data loader test end
    


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
    #             save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_fac1_o_%.5f_%.5f_%.5f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
    #             save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_fac2_o_%.5f_%.5f_%.5f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
    #             save_image(output1[i][0:3].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_out1_o_%.5f_%.5f_%.5f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))
    #             save_image(output1[i][3:6].data,os.path.join(OUTPUT_DIR,"test_interp_gaze_out2_o_%.5f_%.5f_%.5f.png"%(df[i].data,gazeX[i].data,gazeY[i].data)))

    #         # save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fovea_interp_l1_%.5f.png"%(df[0].data)))
    #         # save_image(output[0][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fovea_interp_l2_%.5f.png"%(df[0].data)))
    #         # output = GenRetinalFromLayersBatch(output,conf,df,v,u)
    #         # save_image(output[0][0:3].data,os.path.join(OUTPUT_DIR,"1113_interp_o%.5f.png"%(df[0].data)))
    # exit()

    ################################################ train #########################################################
    if model_type == "RNN": 
        lf_model = model(FIRSST_LAYER_CHANNELS,LAST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,KERNEL_SIZE_RB,INTERLEAVE_RATE)
    else:
        lf_model = model(FIRSST_LAYER_CHANNELS,LAST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,KERNEL_SIZE_RB,INTERLEAVE_RATE,RNN=False)
    lf_model.apply(weight_init_normal)
    lf_model.train()
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
        lf_model = lf_model.to('cuda:2')
    
    optimizer = torch.optim.Adam(lf_model.parameters(),lr=5e-3,betas=(0.9,0.999))
    
    # lf_model.output_layer.register_backward_hook(hook_fn_back)
    if mode=="Perf":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    print("begin training....")
    for epoch in range(epoch_begin, NUM_EPOCH):
        for batch_idx, (image_set, gt,phi, phi_invalid, retinal_invalid, flow, flow_invalid_mask, df, gazeX, gazeY, posX, posY, posZ, sample_idx) in enumerate(train_data_loader):
            # print(sample_idx.shape,df.shape,gazeX.shape,gazeY.shape) # torch.Size([2, 5])
            # print(image_set.shape,gt.shape,gt2.shape) #torch.Size([2, 5, 9, 320, 320, 3]) torch.Size([2, 5, 160, 160, 3]) torch.Size([2, 5, 160, 160, 3])
            # print(delta.shape) # delta: torch.Size([2, 4, 160, 160, 3])
            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("load:",start.elapsed_time(end))

                start.record()
            #reshape for input
            image_set = image_set.permute(0,1,2,5,3,4) # N Seq 5 LF 25 C 3 H W 
            image_set = image_set.reshape(image_set.shape[0],image_set.shape[1],-1,image_set.shape[4],image_set.shape[5]) # N, LFxC, H, W
            # N, Seq 5, LF 25 C 3, H, W 
            image_set = var_or_cuda(image_set)
            # print(image_set.shape) #torch.Size([2, 5, 75, 320, 320])

            gt = gt.permute(0,1,4,2,3) # BS Seq 5 C 3 H W 
            gt = var_or_cuda(gt)

            flow = var_or_cuda(flow) #BS,Seq-1,H,W,2

            # gt2 = gt2.permute(0,1,4,2,3)
            # gt2 = var_or_cuda(gt2)

            gen1 = torch.empty(gt.shape) # BS Seq C H W
            gen1 = var_or_cuda(gen1)
            # print(gen1.shape) #torch.Size([2, 5, 3, 320, 320])

            # gen2 = torch.empty(gt2.shape)
            # gen2 = var_or_cuda(gen2)

            #BS, Seq - 1, C, H, W
            warped = torch.empty(gt.shape[0],gt.shape[1]-1,gt.shape[2],gt.shape[3],gt.shape[4])
            warped = var_or_cuda(warped)
            gen_temp = torch.empty(warped.shape)
            gen_temp = var_or_cuda(gen_temp)
            # print("warped:",warped.shape) #warped: torch.Size([2, 4, 3, 320, 320])
            if mode=="Perf":
                end.record()
                torch.cuda.synchronize()
                print("data prepare:",start.elapsed_time(end))

                start.record()
            if model_type == "RNN": 
                if batch_model != "Single":
                    for k in range(image_set.shape[1]):
                        if k == 0:
                            lf_model.reset_hidden(image_set[:,k])
                        output = lf_model(image_set[:,k],df[:,k],gazeX[:,k],gazeY[:,k],posX[:,k],posY[:,k],posZ[:,k]) # batchsize, layer_num x 2 = 6, layer_res: 320, layer_res: 320
                        output1 = MergeBatchSpeed(output, gen, phi[:,k], phi_invalid[:,k], retinal_invalid[:,k])
                        gen1[:,k] = output1[:,0:3]
                        gt[:,k] = gt[:,k].mul_(~retinal_invalid[:,k].to("cuda:2"))

                        if ((epoch%10 == 0 and epoch != 0) or epoch == 2):
                            for i in range(output.shape[0]):
                                save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fac1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                                save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fac2_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                    
                    for i in range(1,gt.shape[1]):
                        # print(flow_invalid_mask.shape) #torch.Size([2, 4, 320, 320])
                        # print(FlowMap(gen1[:,i-1],flow[:,i-1]).shape) #torch.Size([2, 3, 320, 320])
                        warped[:,i-1] = FlowMap(gen1[:,i-1],flow[:,i-1]).mul(~flow_invalid_mask[:,i-1].unsqueeze(1).to("cuda:2"))
                        gen_temp[:,i-1] = gen1[:,i].mul(~flow_invalid_mask[:,i-1].unsqueeze(1).to("cuda:2"))
                else:
                    for k in range(image_set.shape[1]):
                        if k == 0:
                            lf_model.reset_hidden(image_set[:,k])
                        output = lf_model(image_set[:,k],df[:,k],gazeX[:,k],gazeY[:,k],posX[:,k],posY[:,k],posZ[:,k]) # batchsize, layer_num x 2 = 6, layer_res: 320, layer_res: 320
                        output1 = MergeBatchSpeed(output, gen, phi[:,k], phi_invalid[:,k], retinal_invalid[:,k])
                        gen1[:,k] = output1[:,0:3]
                        gt[:,k] = gt[:,k].mul_(~retinal_invalid[:,k].to("cuda:2"))

                        if k != image_set.shape[1]-1:
                            warped[:,k] = FlowMap(output1.detach(),flow[:,k]).mul(~flow_invalid_mask[:,k].unsqueeze(1).to("cuda:2"))
                        loss1 = loss_without_perc(output1,gt[:,k])
                        loss = (w_frame * loss1)
                        if k==0:
                            loss.backward(retain_graph=False)
                            optimizer.step()
                            lf_model.zero_grad()
                            optimizer.zero_grad()
                            print("Epoch:",epoch,",Iter:",batch_idx,",Seq:",k,",loss:",loss.item())
                        else:
                            output1mask = output1.mul(~flow_invalid_mask[:,k-1].unsqueeze(1).to("cuda:2"))
                            loss2 = l1loss(output1mask,warped[:,k-1])
                            loss += (w_inter_frame * loss2)
                            loss.backward(retain_graph=False)
                            optimizer.step()
                            lf_model.zero_grad()
                            optimizer.zero_grad()
                            # print("Epoch:",epoch,",Iter:",batch_idx,",Seq:",k,",loss:",loss.item())
                            print("Epoch:",epoch,",Iter:",batch_idx,",Seq:",k,",frame loss:",loss1.item(),",inter loss:",w_inter_frame * loss2.item())

                        
                        if ((epoch%10 == 0 and epoch != 0) or epoch == 2):
                            for i in range(output.shape[0]):
                                save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fac1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                                save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fac2_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                    
                    # BSxSeq C H W
                    gen1 = gen1.reshape(-1,gen1.shape[2],gen1.shape[3],gen1.shape[4])
                    gt = gt.reshape(-1,gt.shape[2],gt.shape[3],gt.shape[4])
                    
                    if ((epoch%10== 0 and epoch != 0) or epoch == 2): # torch.Size([2, 5, 160, 160, 3])
                        for i in range(gt.size()[0]):
                            save_image(gen1[i].data,os.path.join(OUTPUT_DIR,"gaze_out1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data,posX[i//5][i%5].data,posY[i//5][i%5].data)))
                            save_image(gt[i].data,os.path.join(OUTPUT_DIR,"gaze_test1_gt0_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data,posX[i//5][i%5].data,posY[i//5][i%5].data)))
                    if ((epoch%100 == 0) and epoch != 0 and batch_idx==len(train_data_loader)-1):
                        save_checkpoints(os.path.join(OUTPUT_DIR, 'gaze-ckpt-epoch-%04d.pth' % (epoch)),epoch,lf_model,optimizer)
            else:
                if batch_model != "Single":
                    for k in range(image_set.shape[1]):
                        output = lf_model(image_set[:,k],df[:,k],gazeX[:,k],gazeY[:,k],posX[:,k],posY[:,k],posZ[:,k]) # batchsize, layer_num x 2 = 6, layer_res: 320, layer_res: 320
                        output1 = MergeBatchSpeed(output, gen, phi[:,k], phi_invalid[:,k], retinal_invalid[:,k])
                        gen1[:,k] = output1[:,0:3]
                        gt[:,k] = gt[:,k].mul_(~retinal_invalid[:,k].to("cuda:2"))
                        
                        if ((epoch%10 == 0 and epoch != 0) or epoch == 2):
                            for i in range(output.shape[0]):
                                save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fac1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                                save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fac2_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                    for i in range(1,gt.shape[1]):
                        # print(flow_invalid_mask.shape) #torch.Size([2, 4, 320, 320])
                        # print(FlowMap(gen1[:,i-1],flow[:,i-1]).shape) #torch.Size([2, 3, 320, 320])
                        warped[:,i-1] = FlowMap(gen1[:,i-1],flow[:,i-1]).mul(~flow_invalid_mask[:,i-1].unsqueeze(1).to("cuda:2"))
                        gen_temp[:,i-1] = gen1[:,i].mul(~flow_invalid_mask[:,i-1].unsqueeze(1).to("cuda:2"))
                else:
                    for k in range(image_set.shape[1]):
                        output = lf_model(image_set[:,k],df[:,k],gazeX[:,k],gazeY[:,k],posX[:,k],posY[:,k],posZ[:,k]) # batchsize, layer_num x 2 = 6, layer_res: 320, layer_res: 320
                        output1 = MergeBatchSpeed(output, gen, phi[:,k], phi_invalid[:,k], retinal_invalid[:,k])
                        gen1[:,k] = output1[:,0:3]
                        gt[:,k] = gt[:,k].mul_(~retinal_invalid[:,k].to("cuda:2"))
                        
                        # print(output1.shape) #torch.Size([BS, 3, 320, 320])
                        loss1 = loss_without_perc(output1,gt[:,k])
                        loss = (w_frame * loss1)
                        # print("loss:",loss1.item())
                        loss.backward(retain_graph=False)
                        optimizer.step()
                        lf_model.zero_grad()
                        optimizer.zero_grad()

                        print("Epoch:",epoch,",Iter:",batch_idx,",Seq:",k,",loss:",loss.item())
                        if ((epoch%10 == 0 and epoch != 0) or epoch == 2):
                            for i in range(output.shape[0]):
                                save_image(output[i][0:3].data,os.path.join(OUTPUT_DIR,"gaze_fac1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                                save_image(output[i][3:6].data,os.path.join(OUTPUT_DIR,"gaze_fac2_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i][k].data,gazeX[i][k].data,gazeY[i][k].data,posX[i][k].data,posY[i][k].data)))
                    for i in range(1,gt.shape[1]):
                        warped[:,i-1] = FlowMap(gen1[:,i-1],flow[:,i-1]).mul(~flow_invalid_mask[:,i-1].unsqueeze(1).to("cuda:2"))
                        gen_temp[:,i-1] = gen1[:,i].mul(~flow_invalid_mask[:,i-1].unsqueeze(1).to("cuda:2"))

                    warped = warped.reshape(-1,warped.shape[2],warped.shape[3],warped.shape[4])
                    gen_temp = gen_temp.reshape(-1,gen_temp.shape[2],gen_temp.shape[3],gen_temp.shape[4])
                    loss3 = l1loss(warped,gen_temp)
                    print("Epoch:",epoch,",Iter:",batch_idx,",inter-frame loss:",w_inter_frame *loss3.item())
                    
                    # BSxSeq C H W
                    gen1 = gen1.reshape(-1,gen1.shape[2],gen1.shape[3],gen1.shape[4])
                    gt = gt.reshape(-1,gt.shape[2],gt.shape[3],gt.shape[4])
                    
                    if ((epoch%10== 0 and epoch != 0) or epoch == 2): # torch.Size([2, 5, 160, 160, 3])
                        for i in range(gt.size()[0]):
                            save_image(gen1[i].data,os.path.join(OUTPUT_DIR,"gaze_out1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data,posX[i//5][i%5].data,posY[i//5][i%5].data)))
                            save_image(gt[i].data,os.path.join(OUTPUT_DIR,"gaze_test1_gt0_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data,posX[i//5][i%5].data,posY[i//5][i%5].data)))
                    if ((epoch%100 == 0) and epoch != 0 and batch_idx==len(train_data_loader)-1):
                        save_checkpoints(os.path.join(OUTPUT_DIR, 'gaze-ckpt-epoch-%04d.pth' % (epoch)),epoch,lf_model,optimizer)


            if batch_model != "Single":
                if mode=="Perf":
                    end.record()
                    torch.cuda.synchronize()
                    print("forward:",start.elapsed_time(end))

                    start.record()
                optimizer.zero_grad()

                
                # BSxSeq C H W
                gen1 = gen1.reshape(-1,gen1.shape[2],gen1.shape[3],gen1.shape[4])
                # gen2 = gen2.reshape(-1,gen2.shape[2],gen2.shape[3],gen2.shape[4])
                # BSxSeq C H W
                gt = gt.reshape(-1,gt.shape[2],gt.shape[3],gt.shape[4])
                # gt2 = gt2.reshape(-1,gt2.shape[2],gt2.shape[3],gt2.shape[4])

                # BSx(Seq-1) C H W
                warped = warped.reshape(-1,warped.shape[2],warped.shape[3],warped.shape[4])
                gen_temp = gen_temp.reshape(-1,gen_temp.shape[2],gen_temp.shape[3],gen_temp.shape[4])
                
                loss1_value = loss1(gen1,gt)
                loss2_value = loss2(warped,gen_temp)
                if model_type == "RNN": 
                    loss = (w_frame * loss1_value)+ (w_inter_frame * loss2_value)
                else:
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

                print("Epoch:",epoch,",Iter:",batch_idx,",loss:",loss.item(),",frame loss:",loss1_value.item(),",inter-frame loss:",loss2_value.item())
                
                # exit(0)
                ########################### Save #####################
                if ((epoch%10== 0 and epoch != 0) or epoch == 2): # torch.Size([2, 5, 160, 160, 3])
                    for i in range(gt.size()[0]):
                        # df 2,5 
                        save_image(gen1[i].data,os.path.join(OUTPUT_DIR,"gaze_out1_o_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data,posX[i//5][i%5].data,posY[i//5][i%5].data)))
                        # save_image(gen2[i].data,os.path.join(OUTPUT_DIR,"gaze_out2_o_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data)))
                        save_image(gt[i].data,os.path.join(OUTPUT_DIR,"gaze_test1_gt0_%.5f_%.5f_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data,posX[i//5][i%5].data,posY[i//5][i%5].data)))
                        # save_image(gt2[i].data,os.path.join(OUTPUT_DIR,"gaze_test1_gt1_%.5f_%.5f_%.5f.png"%(df[i//5][i%5].data,gazeX[i//5][i%5].data,gazeY[i//5][i%5].data)))
                if ((epoch%100 == 0) and epoch != 0 and batch_idx==len(train_data_loader)-1):
                    save_checkpoints(os.path.join(OUTPUT_DIR, 'gaze-ckpt-epoch-%04d.pth' % (epoch)),epoch,lf_model,optimizer)