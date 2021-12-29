from __future__ import division
import os.path
from .listdataset import  ListDataset

import numpy as np
import random
import flow_transforms
import pdb
import glob
from tqdm import tqdm
from tqdm import trange

try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

def index_items(sample_list, train_index, val_index):
    train_samples = [sample_list[idx] for idx in train_index]
    val_samples = [sample_list[idx] for idx in val_index]

    return train_samples, val_samples

def make_dataset(path):
    # we train and val seperately to tune the hyper-param and use all the data for the final training
    #train_list_path = os.path.join(dir, 'train.txt') # use train_Val.txt for final report
    #val_list_path = os.path.join(dir, 'val.txt')
    gt_nii_img = glob.glob(path + '*GG/seg/*nii.gz')
    t1_nii_img = [item.replace('/seg/','/t1/').replace('_seg.', '_t1.') for item in gt_nii_img]
    t1ce_nii_img = [item.replace('/seg/','/t1ce/').replace('_seg.', '_t1ce.') for item in gt_nii_img]
    t2_nii_img = [item.replace('/seg/','/t2/').replace('_seg.', '_t2.') for item in gt_nii_img]
    flair_nii_img = [item.replace('/seg/','/flair/').replace('_seg.', '_flair.') for item in gt_nii_img]

    total_index = list(range(len(gt_nii_img)))
    #total_index = list(range(20))

    random.shuffle(total_index)
    train_num = int(len(total_index) * 0.75*0.85) 
    val_num = int(len(total_index) * 0.75*0.15) 
    train_index = total_index[:train_num]
    val_index = total_index[train_num:(train_num+val_num)]

    train_gt_list, val_gt_list = index_items(gt_nii_img, train_index, val_index)
    train_t1_list, val_t1_list = index_items(t1_nii_img, train_index, val_index) 
    train_t1ce_list, val_t1ce_list = index_items(t1ce_nii_img, train_index, val_index) 
    train_t2_list, val_t2_list = index_items(t2_nii_img, train_index, val_index) 
    train_flair_list, val_flair_list = index_items(flair_nii_img, train_index, val_index) 
   
    train_dict = {
        't1':train_t1_list,
        't1ce':train_t1ce_list,
        't2':train_t2_list,
        'flair':train_flair_list,
        'gt':train_gt_list
        } 

    val_dict = {
        't1':val_t1_list,
        't1ce':val_t1ce_list,
        't2':val_t2_list,
        'flair':val_flair_list,
        'gt':val_gt_list
        } 
    return train_dict, val_dict

def BSD_loader(data_dict):
    # cv2.imread is faster than io.imread usually
    nonzero_index = []
    def pick_slice(seg_im):
        h,w,slice_num = seg_im.shape
        seg_im = np.reshape(seg_im, (-1, slice_num))
        sum_value  = np.sum(seg_im, axis=0)
        nonzero_index = np.where(sum_value > 0)

        return nonzero_index[0]
        
    seg_list = data_dict['gt']
    t1_list = data_dict['t1']
    t1ce_list = data_dict['t1ce']
    t2_list = data_dict['t2']
    flair_list = data_dict['flair']
    
    nii_num = len(seg_list)
    im_list = []
    gt_list = []
    for i in trange(nii_num):
        seg_im = nib.load(seg_list[i]).get_fdata()
        nonzero_index = pick_slice(seg_im)
        seg_slice = seg_im[:,:,nonzero_index]
        seg_slice = np.transpose(seg_slice, (2, 0, 1)).astype(np.uint8)

        t1_im = nib.load(t1_list[i]).get_fdata()
        t1_slice = t1_im[:,:,nonzero_index]
        t1_slice = np.transpose(t1_slice, (2, 0, 1))
        #t1_slice = (t1_slice / np.max(t1_slice) * 255).astype(np.uint8)
 
        t1ce_im = nib.load(t1ce_list[i]).get_fdata()
        t1ce_slice = t1ce_im[:,:,nonzero_index]
        t1ce_slice = np.transpose(t1ce_slice, (2, 0, 1))
        #t1ce_slice = (t1ce_slice / np.max(t1ce_slice) * 255).astype(np.uint8)
        
        t2_im = nib.load(t2_list[i]).get_fdata()
        t2_slice = t2_im[:,:,nonzero_index]
        t2_slice = np.transpose(t2_slice, (2, 0, 1))
        #t2_slice = (t2_slice / np.max(t2_slice) * 255).astype(np.uint8)
 
        flair_im = nib.load(flair_list[i]).get_fdata()
        flair_slice = flair_im[:,:,nonzero_index]
        flair_slice = np.transpose(flair_slice, (2, 0, 1))
        #flair_slice = (flair_slice / np.max(flair_slice) * 255).astype(np.uint8)

        img_data = np.concatenate([t1_slice[:,:,:,np.newaxis], t1ce_slice[:,:,:,np.newaxis], t2_slice[:,:,:,np.newaxis], flair_slice[:,:,:,np.newaxis]], axis=-1)
        img_data = (img_data / np.max(img_data)*255).astype(np.uint8)

        im_list.append(img_data)
        gt_list.append(seg_slice[:,:,:,np.newaxis])

    im_data = np.concatenate(im_list, axis=0)
    gt_data = np.concatenate(gt_list, axis=0)

    brain = (im_data[:,:,:,:1] > 0) * 3
    mask = 1 - (gt_data > 0) * 1
    normal_brain = brain * mask
    gt_data = normal_brain.astype(np.uint8) + gt_data.astype(np.uint8)
 
    return im_data, gt_data
    #return np.transpose(im_data, (0,3,1,2)), np.transpose(gt_data, (0,3,1,2))

def BSD500(root, transform=None, target_transform=None, val_transform=None,
              co_transform=None, split=None):
    train_dict, val_dict= make_dataset(root)

    if val_transform ==None:
        val_transform = transform

    train_dataset = ListDataset(root, 'bsd500', train_dict, transform,
                                target_transform, co_transform,
                                loader=BSD_loader, datatype = 'train')

    val_dataset = ListDataset(root, 'bsd500', val_dict, val_transform,
                               target_transform, flow_transforms.CenterCrop((240,240)),
                               loader=BSD_loader, datatype = 'val')

    return train_dataset, val_dataset

def mix_datasets(args):
    train_dataset = ListDataset(args, 'train')
    val_dataset = ListDataset(args, 'val')

    return train_dataset, val_dataset
