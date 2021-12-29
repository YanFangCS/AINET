import torch.utils.data as data
import pdb
import torch
import numpy as np
import random
import cv2
import torchvision.transforms as transforms
import flow_transforms
from .dataset_loader import *
import os
import glob
from tqdm import trange
import json
#from .BSD500 import make_dataset

class ListDataset(data.Dataset):
    #def __init__(self, root, dataset, data_dict, transform=None, target_transform=None,
    #             co_transform=None, loader=None, datatype=None):
    def __init__(self, args, flag):
        self.data_root = args.data
        self.target_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
        ])
        self.debug_num = 20 
        self.debug = args.debug
        self.data_length = 0
        self.flag = flag
        
        self.BDS_list = open(self.data_root + f'{flag}.txt').readlines()
        #self.BDS_list = self.BDS_list[:self.debug_num]
        self.BDS_ttrans,self.BDS_coTrans = get_transform(args, dataset='BDS500', flag=flag)
        #if flag == 'train':
        #    random.shuffle(self.BDS_list)

        if self.debug:
            self.BDS_list = self.BDS_list[:self.debug_num]

        self.BDS_length = len(self.BDS_list)
        if len(self.BDS_list) > self.data_length:
            self.data_length = len(self.BDS_list)

    def __getitem__(self, index):
        data = []
        im_path = self.BDS_list[index]
        bds_im, bds_label, patch_posi, patch_label = BDS500(im_path, self.BDS_ttrans, self.BDS_coTrans, self.target_transform, self.flag)
        data.append(bds_im)
        data.append(bds_label) 
        data.append(patch_posi) 
        data.append(patch_label)

        return data

    def __len__(self):
        return self.data_length


