import numpy as np
from PIL import Image
import flow_transforms
import torchvision.transforms as transforms
import torch
import cv2
from .BSD500 import *
import pdb
from tqdm import trange
import glob
import random
import pdb
from .data_util import local_patch_sampler, patch_shuffle 
import sys
#

def BDS500(path_imgs, input_transform, co_transform, target_transform, flag):
    path_imgs = path_imgs.strip()
    path_label = path_imgs.replace('_img.jpg', '_label.png')
    img = cv2.imread(path_imgs)[:, :, ::-1].astype(np.float32)
    #img = np.concatenate([img, img[:,:,-1:]], axis=-1)
    gtseg = cv2.imread(path_label)[:,:,:1]
    assert np.max(gtseg) <= 50 and np.min(gtseg) >= 0 
    #conduct transform
    img, gtseg = co_transform([img], gtseg)
    '''
    if np.unique(gtseg).size==1:
        pdb.set_trace()
        print(path_imgs)
        cv2.imwrite('im.png', img[0])
        cv2.imwrite('label.png', gtseg)
        sys.exit(0)
        img, gtseg = co_transform(img, gtseg)
    '''
    
    patch_labels, patch_posis = local_patch_sampler(gtseg)
    image = input_transform(img[0])
    label = target_transform(gtseg)

    #patch_posis = patch_posis[:,:,None]
    patch_posis = torch.from_numpy(patch_posis).long()
    patch_labels = torch.from_numpy(patch_labels).float()

    if flag == 'train':
        image, label = patch_shuffle(image, label)
        image, label = patch_shuffle(image, label)

    return image, label, patch_posis, patch_labels

def get_transform(args, dataset, flag):
    crop_shape = (args.train_img_height, args.train_img_width)
    mean = args.dataset_mean
    
    val_crop = (208,208)
    mean1 = [0,0,0]
    std1=[255,255,255]
    std2=[1,1,1]
                        
    if flag == 'train':
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=mean1, std=std1),
            transforms.Normalize(mean=mean, std=std2)
        ])
        co_transform = flow_transforms.Compose([
                flow_transforms.RandomCrop(crop_shape),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip()])
    else:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean = mean1, std=std1),
            transforms.Normalize(mean = mean, std=std2)
        ])
        co_transform = flow_transforms.Compose([
                flow_transforms.CenterCrop(val_crop)
            ])

    return input_transform, co_transform
