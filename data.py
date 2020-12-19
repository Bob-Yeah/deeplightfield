import torch
import os
import glob
import numpy as np
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader 

import cv2
import json
from Flow import *
from gen_image import *
import util

import time


class lightFieldSynDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path,file_json):
        self.file_dir_path = file_dir_path
        with open(file_json, encoding='utf-8') as file:
            self.dataset_desc = json.loads(file.read())
        self.input_img = []
        for i in self.dataset_desc["train"]:
            lf_element = os.path.join(self.file_dir_path,i)
            lf_element = cv2.imread(lf_element, -cv2.IMREAD_ANYDEPTH)[:, :, 0:3]
            lf_element = cv2.cvtColor(lf_element, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            lf_element = lf_element[:,:-1,:]
            self.input_img.append(lf_element)
        self.input_img = np.asarray(self.input_img)
    def __len__(self):
        return len(self.dataset_desc["gt"])

    def __getitem__(self, idx):
        gt, pos_row, pos_col  = self.get_datum(idx)
        return (self.input_img, gt, pos_row, pos_col)

    def get_datum(self, idx):
        fd_gt_path = os.path.join(self.file_dir_path, self.dataset_desc["gt"][idx])
        gt = cv2.imread(fd_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        gt = gt[:,:-1,:]
        pos_col = self.dataset_desc["x"][idx]
        pos_row = self.dataset_desc["y"][idx]
        return gt, pos_row, pos_col

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
        lf_image_paths = os.path.join(self.file_dir_path, self.dataset_desc["train"][idx])
        # print(lf_image_paths)
        fd_gt_path = os.path.join(self.file_dir_path, self.dataset_desc["gt"][idx])
        fd_gt_path2 = os.path.join(self.file_dir_path, self.dataset_desc["gt2"][idx])
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
        lf_image_paths = os.path.join(self.file_dir_path, self.dataset_desc["train"][idx])
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
            lf_image_paths = os.path.join(self.file_dir_path, self.dataset_desc["train"][indices[i]])
            fd_gt_path = os.path.join(self.file_dir_path, self.dataset_desc["gt"][indices[i]])
            fd_gt_path2 = os.path.join(self.file_dir_path, self.dataset_desc["gt2"][indices[i]])
            lf_image_one_sample = []
            lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)

            for j in range(9):
                lf_image = lf_image_big[j//3*IM_H:j//3*IM_H+IM_H,j%3*IM_W:j%3*IM_W+IM_W,0:3]
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

class lightFieldFlowSeqDataLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir_path, file_json, gen, conf, transforms=None):
        self.file_dir_path = file_dir_path
        self.file_json = file_json
        self.gen = gen
        self.conf = conf
        self.transforms = transforms
        with open(file_json, encoding='utf-8') as file:
            self.dataset_desc = json.loads(file.read())

    def __len__(self):
        return len(self.dataset_desc["seq"])

    def __getitem__(self, idx):
        # start = time.time()
        lightfield_images, gt, phi, phi_invalid, retinal_invalid, flow, flow_invalid_mask, fd, gazeX, gazeY, posX, posY, posZ, sample_idx = self.get_datum(idx)
        fd = fd.astype(np.float32)
        gazeX = gazeX.astype(np.float32)
        gazeY = gazeY.astype(np.float32)
        posX = posX.astype(np.float32)
        posY = posY.astype(np.float32)
        posZ = posZ.astype(np.float32)
        sample_idx = sample_idx.astype(np.int64)

        if self.transforms:
            lightfield_images = self.transforms(lightfield_images)
        # print("read once:",time.time() - start) # 8 ms
        return (lightfield_images, gt, phi, phi_invalid, retinal_invalid, flow, flow_invalid_mask, fd, gazeX, gazeY, posX, posY, posZ, sample_idx)

    def get_datum(self, idx):
        IM_H = 320
        IM_W = 320
        indices = self.dataset_desc["seq"][idx]
        # print("indices:",indices)
        lf_images = []
        fd = []
        gazeX = []
        gazeY = []
        posX = []
        posY = []
        posZ = []
        sample_idx = []
        gt = []
        # gt2 = []
        phi = []
        phi_invalid = []
        retinal_invalid = []
        for i in range(len(indices)): # 5
            lf_image_paths = os.path.join(self.file_dir_path, self.dataset_desc["train"][indices[i]])
            fd_gt_path = os.path.join(self.file_dir_path, self.dataset_desc["gt"][indices[i]])
            # fd_gt_path2 = os.path.join(self.file_dir_path, self.dataset_desc["gt2"][indices[i]])
            lf_image_one_sample = []
            lf_image_big = cv2.imread(lf_image_paths, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            lf_image_big = cv2.cvtColor(lf_image_big,cv2.COLOR_BGR2RGB)

            lf_dim = int(self.conf.light_field_dim)
            for j in range(lf_dim**2):
                lf_image = lf_image_big[j//lf_dim*IM_H:j//lf_dim*IM_H+IM_H,j%lf_dim*IM_W:j%lf_dim*IM_W+IM_W,0:3]
                lf_image_one_sample.append(lf_image)

            gt_i = cv2.imread(fd_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            gt.append(cv2.cvtColor(gt_i,cv2.COLOR_BGR2RGB))
            # gt2_i = cv2.imread(fd_gt_path2, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            # gt2.append(cv2.cvtColor(gt2_i,cv2.COLOR_BGR2RGB))

            # print("indices[i]:",indices[i])
            fd.append([self.dataset_desc["focaldepth"][indices[i]]])
            gazeX.append([self.dataset_desc["gazeX"][indices[i]]])
            gazeY.append([self.dataset_desc["gazeY"][indices[i]]])
            posX.append([self.dataset_desc["x"][indices[i]]])
            posY.append([self.dataset_desc["y"][indices[i]]])
            posZ.append([0.0])
            sample_idx.append([self.dataset_desc["idx"][indices[i]]])
            lf_images.append(lf_image_one_sample)

            idx_i = sample_idx[i][0]
            focaldepth_i = fd[i][0]
            gazeX_i = gazeX[i][0]
            gazeY_i = gazeY[i][0]
            posX_i = posX[i][0]
            posY_i = posY[i][0]
            posZ_i = posZ[i][0]
            # print("x:%.3f,y:%.3f,z:%.3f;gaze:%.4f,%.4f,focaldepth:%.3f."%(posX_i,posY_i,posZ_i,gazeX_i,gazeY_i,focaldepth_i))
            phi_i,phi_invalid_i,retinal_invalid_i =  self.gen.CalculateRetinal2LayerMappings(torch.tensor([posX_i, posY_i,posZ_i]),torch.tensor([gazeX_i, gazeY_i]),focaldepth_i)

            phi.append(phi_i)
            phi_invalid.append(phi_invalid_i)
            retinal_invalid.append(retinal_invalid_i)
        #lf_images: 5,9,320,320
        flow = Flow.Load([os.path.join(self.file_dir_path, self.dataset_desc["flow"][indices[i-1]]) for i in range(1,len(indices))])
        flow_map = flow.getMap()
        flow_invalid_mask = flow.b_invalid_mask
        # print("flow:",flow_map.shape)

        return np.asarray(lf_images),np.asarray(gt), torch.stack(phi,dim=0), torch.stack(phi_invalid,dim=0),torch.stack(retinal_invalid,dim=0), flow_map, flow_invalid_mask, np.asarray(fd),np.asarray(gazeX),np.asarray(gazeY),np.asarray(posX),np.asarray(posY),np.asarray(posZ),np.asarray(sample_idx)
