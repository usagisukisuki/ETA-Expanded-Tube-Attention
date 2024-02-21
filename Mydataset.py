from __future__ import print_function
import torch
import torch.utils.data as data
import os
import random
import glob
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as tvf
from PIL import Image, ImageOps
import cv2

class Dataset_ETA(data.Dataset):
    def __init__(self, dataset='DRIVE', root=None, cross=None, dataset_type='train',transform=None, K=8):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.root = root
        self.transform = transform
        
        if self.dataset_type == 'train':
            self.root_img = sorted(os.listdir(root + "/{}/HR".format(self.dataset)))
            self.root_gt = sorted(os.listdir(root + "/{}/VE".format(self.dataset)))
             
        elif self.dataset_type == 'test':
            self.root_img = sorted(os.listdir(root + "/{}T/HR".format(self.dataset)))
            self.root_gt = sorted(os.listdir(root + "/{}T/VE".format(self.dataset)))


        ### mask kernel ###
        self.kernel1 = np.ones((3,3),np.uint8)
        self.kernel2 = np.ones((5,5),np.uint8)
        self.kernel3 = np.ones((7,7),np.uint8)
        

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            img_name = self.root + "/{}/HR/".format(self.dataset) + self.root_img[index]
            label_name = self.root + "/{}/VE/".format(self.dataset) + self.root_gt[index]
        elif self.dataset_type == 'test':
            img_name = self.root + "/{}T/HR/".format(self.dataset) + self.root_img[index]
            label_name = self.root + "/{}T/VE/".format(self.dataset) + self.root_gt[index]
            
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")
        image = np.array(image)
        mask = np.array(label)
        mask = np.where(mask>150, 1, mask)
        mask = np.uint8(mask)
        
        if self.dataset_type == 'train':
            mask1 = cv2.dilate(mask, self.kernel3, iterations = 1)
            mask2 = cv2.dilate(mask, self.kernel2, iterations = 1)
            mask3 = cv2.dilate(mask, self.kernel1, iterations = 1)
        
            #mask = np.eye(2)[mask]
            image = Image.fromarray(np.uint8(image))
            mask1 = Image.fromarray(np.uint8(mask1))
            mask2 = Image.fromarray(np.uint8(mask2))
            mask3 = Image.fromarray(np.uint8(mask3))
            mask = Image.fromarray(mask)

            if self.transform:
                image, mask1, mask2, mask3, mask = self.transform(image, mask1, mask2, mask3, mask)

            return image, mask1, mask2, mask3, mask
            
        elif self.dataset_type == 'test':
            image = Image.fromarray(np.uint8(image))
            mask = Image.fromarray(mask)

            if self.transform:
                image, mask = self.transform(image, mask)

            return image, mask

    def __len__(self):
        return len(self.root_img)
      
        
class Dataset_ETA_test(data.Dataset):
    def __init__(self, dataset='DRIVE', root=None, cross=None, dataset_type='train',transform=None, K=8):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.root = root
        self.transform = transform
        
        if self.dataset_type == 'train':
            self.root_img = sorted(os.listdir(root + "/{}/HR".format(self.dataset)))
            self.root_gt = sorted(os.listdir(root + "/{}/VE".format(self.dataset)))
             
        elif self.dataset_type == 'test':
            self.root_img = sorted(os.listdir(root + "/{}T/HR".format(self.dataset)))
            self.root_gt = sorted(os.listdir(root + "/{}T/VE".format(self.dataset)))

        
        ### mask kernel ###
        self.kernel1 = np.ones((3,3),np.uint8)
        self.kernel2 = np.ones((5,5),np.uint8)
        self.kernel3 = np.ones((7,7),np.uint8)
        

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            img_name = self.root + "/{}/HR/".format(self.dataset) + self.root_img[index]
            label_name = self.root + "/{}/VE/".format(self.dataset) + self.root_gt[index]
        elif self.dataset_type == 'test':
            img_name = self.root + "/{}T/HR/".format(self.dataset) + self.root_img[index]
            label_name = self.root + "/{}T/VE/".format(self.dataset) + self.root_gt[index]
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")
        image = np.array(image)
        mask = np.array(label)
        mask = np.where(mask>150, 1, mask)
        mask = np.uint8(mask)
        
        mask1 = cv2.dilate(mask, self.kernel3, iterations = 1)
        mask2 = cv2.dilate(mask, self.kernel2, iterations = 1)
        mask3 = cv2.dilate(mask, self.kernel1, iterations = 1)
        
        image = Image.fromarray(np.uint8(image))
        mask1 = Image.fromarray(np.uint8(mask1))
        mask2 = Image.fromarray(np.uint8(mask2))
        mask3 = Image.fromarray(np.uint8(mask3))
        mask = Image.fromarray(mask)

        if self.transform:
            image, mask1, mask2, mask3, mask = self.transform(image, mask1, mask2, mask3, mask)

        return image, mask1, mask2, mask3, mask


    def __len__(self):
        return len(self.root_img)


