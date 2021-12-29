import torch
import random
import copy
import numpy as np
import pdb
import cv2
import os

def patch_shuffle(image_data, label_data, region_size=16):
    #shuffle_flag = random.uniform(0,1.)
    shuffle_flag = np.random.rand()
    if shuffle_flag > 0.5:
        return image_data, label_data
    
    c, h, w = image_data.shape
    x_interval =  h // region_size - 1
    y_interval = w // region_size - 1

    x_index1 = random.randint(0, x_interval) * 16
    y_index1 = random.randint(0, y_interval) * 16

    x_index2 = random.randint(0, x_interval) * 16
    y_index2 = random.randint(0, y_interval) * 16

    while x_index1==x_index2 and y_index1==y_index2:
        x_index2 = random.randint(0, x_interval)
        y_index2 = random.randint(0, y_interval)

    image = copy.deepcopy(image_data)
    label = copy.deepcopy(label_data)

    im_patch1 = image[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size]
    im_patch2 = image[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size]

    gt_patch1 = label[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size]
    gt_patch2 = label[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size]
    
    image_data[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size] = im_patch2
    image_data[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size] = im_patch1

    label_data[:,x_index1:x_index1+region_size, y_index1:y_index1+region_size] = gt_patch2
    label_data[:,x_index2:x_index2+region_size, y_index2:y_index2+region_size] = gt_patch1

    image_data, label_data = random_offset(image_data, label_data, x_index1, x_index2, y_index1, y_index2, x_interval, y_interval)
    #pdb.set_trace()
    #cv2.imwrite('im.png', (image.permute(1,2,0).numpy()[:,:,::-1] + 0.5)*255)
    #cv2.imwrite('im_sf.png', (image_data.permute(1,2,0).numpy()[:,:,::-1] + 0.5)*255)
    #cv2.imwrite('label_sf.png', (label_data.permute(1,2,0).numpy()[:,:,0])*50)
    #pdb.set_trace()

    return image_data, label_data

def random_offset(image_data, label_data, x_index1, x_index2, y_index1, y_index2, x_interval, y_interval):
    h_or_v_flag = np.random.rand() #determinte which direction to conduct offset
    H_offset = h_or_v_flag > 0.5
    region_size = 16
    offset_dis = random.randint(0,16)
    if offset_dis == 0:
        return image_data, label_data

    if H_offset:
        #random offset along horizon direction
        x_idx = random.randint(0, x_interval) * 16
        start_idx = random.randint(0, y_interval)
        end_idx = random.randint(start_idx + 1, y_interval+1) * 16
        start_idx = start_idx * 16

        im_patch = image_data[:, x_idx:x_idx+region_size, start_idx:end_idx]
        gt_patch = label_data[:,x_idx:x_idx+region_size, start_idx:end_idx]
        #patch_len = end_idx - start_idx

        replace_or_zero = np.random.rand()
        if replace_or_zero > 0.5: #replace
            if replace_or_zero > 0.75:#forward 
                new_im_patch = torch.cat([im_patch[:, :, -offset_dis:], im_patch[:, :, :-offset_dis]], dim=2)
                new_gt_patch = torch.cat([gt_patch[:, :, -offset_dis:], gt_patch[:, :, :-offset_dis]], dim=2)
            else:#backward
                new_im_patch = torch.cat([im_patch[:,:, offset_dis:], im_patch[:, :, :offset_dis]], dim=2)
                new_gt_patch = torch.cat([gt_patch[:,:, offset_dis:], gt_patch[:, :, :offset_dis]], dim=2)

            image_data[:, x_idx:x_idx+region_size, start_idx:end_idx] = new_im_patch 
            label_data[:,x_idx:x_idx+region_size, start_idx:end_idx] = new_gt_patch
        else: #random fill
            random_im_patch = torch.rand(3, 16, offset_dis) * 2 -1 # to -1---1
            random_gt_patch = torch.ones(1,16,offset_dis) * 50
            if replace_or_zero < 0.25:#forward 
                new_im_patch = torch.cat([random_im_patch, im_patch[:, :, :-offset_dis]], dim=2)
                new_gt_patch = torch.cat([random_gt_patch, gt_patch[:, :, :-offset_dis]], dim=2)
            else:#backward
                new_im_patch = torch.cat([im_patch[:,:, offset_dis:], random_im_patch], dim=2)
                new_gt_patch = torch.cat([gt_patch[:,:, offset_dis:], random_gt_patch], dim=2)

            image_data[:, x_idx:x_idx+region_size, start_idx:end_idx] = new_im_patch 
            label_data[:,x_idx:x_idx+region_size, start_idx:end_idx] = new_gt_patch
    else:
        #random offset along horizon direction
        y_idx = random.randint(0, y_interval) * 16
        start_idx = random.randint(0, x_interval)
        end_idx = random.randint(start_idx + 1, x_interval+1) * 16
        start_idx = start_idx * 16

        im_patch = image_data[:, start_idx:end_idx, y_idx:y_idx+region_size]
        gt_patch = label_data[:, start_idx:end_idx, y_idx:y_idx+region_size]
        patch_len = end_idx - start_idx

        replace_or_zero = np.random.rand()
        if replace_or_zero > 0.5: #replace
            if replace_or_zero > 0.75:#forward 
                new_im_patch = torch.cat([im_patch[:,-offset_dis:, :], im_patch[:, :-offset_dis,:]], dim=1)
                new_gt_patch = torch.cat([gt_patch[:,-offset_dis:, :], gt_patch[:, :-offset_dis,:]], dim=1)
            else:#backward
                new_im_patch = torch.cat([im_patch[:,offset_dis:,:], im_patch[:, :offset_dis, :]], dim=1)
                new_gt_patch = torch.cat([gt_patch[:,offset_dis:,:], gt_patch[:, :offset_dis, :]], dim=1)

            image_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_im_patch 
            label_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_gt_patch
        else: #random fill
            random_im_patch = torch.rand(3, offset_dis, 16) * 2 -1
            random_gt_patch = torch.ones(1,offset_dis, 16) * 50
            if replace_or_zero < 0.25:#forward 
                new_im_patch = torch.cat([random_im_patch, im_patch[:, :-offset_dis, :]], dim=1)
                new_gt_patch = torch.cat([random_gt_patch, gt_patch[:, :-offset_dis,:]], dim=1)
            else:#backward
                new_im_patch = torch.cat([im_patch[:,offset_dis:, :], random_im_patch], dim=1)
                new_gt_patch = torch.cat([gt_patch[:,offset_dis:, :], random_gt_patch], dim=1)
            
            image_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_im_patch 
            label_data[:, start_idx:end_idx, y_idx:y_idx+region_size] = new_gt_patch

    return image_data, label_data

            
