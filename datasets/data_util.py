import numpy as np
import os
from skimage.segmentation import find_boundaries
import random
from PIL import Image
import torch
import sys
import cv2
import pdb

def patch_shuffle(image_data, label_data, region_size=16):
    #shuffle_flag = random.uniform(0,1.)
    shuffle_flag = np.random.rand()
    if shuffle_flag > 0.5:
        return image_data, label_data
    
    c, h, w = image_data.shape
    x_interval =  h // region_size - 1
    y_interval = w // region_size - 1

    x_index1 = random.randint(0, x_interval*16)# * 16
    y_index1 = random.randint(0, y_interval*16)# * 16

    x_index2 = random.randint(0, x_interval*16) # * 16
    y_index2 = random.randint(0, y_interval*16) #* 16

    while x_index1==x_index2 and y_index1==y_index2:
        x_index2 = random.randint(0, x_interval*16)
        y_index2 = random.randint(0, y_interval*16) #*16

    #image = copy.deepcopy(image_data)
    #label = copy.deepcopy(label_data)
    image = image_data
    label = label_data

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
        x_idx = random.randint(0, x_interval*16) #* 16
        start_idx = random.randint(0, y_interval*16)
        end_idx = random.randint(start_idx + 16, (y_interval+1)*16) #* 16
        #start_idx = start_idx * 16

        im_patch = image_data[:, x_idx:x_idx+region_size, start_idx:end_idx]
        gt_patch = label_data[:,x_idx:x_idx+region_size, start_idx:end_idx]
        #patch_len = end_idx - start_idx

        replace_or_zero = np.random.rand()
        if replace_or_zero > 0.5: #replace
            #if replace_or_zero > 0.75:#forward 
            bf = int((replace_or_zero>0.75)*2-1)
            new_im_patch = torch.cat([im_patch[:, :, -bf*offset_dis:], im_patch[:, :, :-bf*offset_dis]], dim=2)
            new_gt_patch = torch.cat([gt_patch[:, :, -bf*offset_dis:], gt_patch[:, :, :-bf*offset_dis]], dim=2)
            #else:#backward
            #    new_im_patch = torch.cat([im_patch[:,:, offset_dis:], im_patch[:, :, :offset_dis]], dim=2)
            #    new_gt_patch = torch.cat([gt_patch[:,:, offset_dis:], gt_patch[:, :, :offset_dis]], dim=2)

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
        y_idx = random.randint(0, y_interval*16) #* 16
        start_idx = random.randint(0, x_interval*16)
        end_idx = random.randint(start_idx + 16, (x_interval+1)*16)
        #start_idx = start_idx * 16

        im_patch = image_data[:, start_idx:end_idx, y_idx:y_idx+region_size]
        gt_patch = label_data[:, start_idx:end_idx, y_idx:y_idx+region_size]
        patch_len = end_idx - start_idx

        replace_or_zero = np.random.rand()
        if replace_or_zero > 0.5: #replace
            #if replace_or_zero > 0.75:#forward 
            bf = int((replace_or_zero>0.75)*2-1)
            new_im_patch = torch.cat([im_patch[:,-bf*offset_dis:, :], im_patch[:, :-bf*offset_dis,:]], dim=1)
            new_gt_patch = torch.cat([gt_patch[:,-bf*offset_dis:, :], gt_patch[:, :-bf*offset_dis,:]], dim=1)
            #else:#backward
            #    new_im_patch = torch.cat([im_patch[:,offset_dis:,:], im_patch[:, :offset_dis, :]], dim=1)
            #    new_gt_patch = torch.cat([gt_patch[:,offset_dis:,:], gt_patch[:, :offset_dis, :]], dim=1)

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


def select_label(label_patch):
    labels = np.unique(label_patch)
    index1 = np.where(label_patch==labels[0])
    index2 = np.where(label_patch==labels[1])
    size1 = index1[0].size
    size2 = index2[0].size

    patch_label1_1 = np.zeros_like(label_patch)
    patch_label1_2 = np.zeros_like(label_patch)
    index1_1 = (index1[0][:size1//2], index1[1][:size1//2])
    index1_2 = (index1[0][size1//2:], index1[1][size1//2:])
    patch_label1_1[index1_1] = 1
    patch_label1_2[index1_2] = 1

    patch_label2_1 = np.zeros_like(label_patch)
    patch_label2_2 = np.zeros_like(label_patch)
    index2_1 = (index2[0][:size2//2], index2[1][:size2//2])
    index2_2 = (index2[0][size2//2:], index2[1][size2//2:])
    patch_label2_1[index2_1] = 1
    patch_label2_2[index2_2] = 1

    patchs = np.concatenate([patch_label1_1[None,:,:], patch_label1_2[None, :, :], patch_label2_1[None, :, :], patch_label2_2[None,:,:]], axis=0)

    return patchs

def local_patch_sampler(seg_label, patch_height=5, patch_width=5, disc=1, max_patch=500):
    seg_label = seg_label[:,:,0]
    seg_boundaries = find_boundaries(seg_label) * 1
    #determine the patch number
    patch_num = np.sum(seg_boundaries) // (max(patch_height, patch_width)*disc)
    patch_num = min(max_patch, patch_num)

    seg_boundaries[0:patch_height,:] = 0
    seg_boundaries[:, 0:patch_width] = 0
    seg_boundaries[-patch_height:,:] =0
    seg_boundaries[:, -patch_width:] =0

    bd_index = np.where(seg_boundaries==1)
    total_bs_pixels = bd_index[0].size

    patch_list = [] # record the patch posi, idx and offset
    label_list = []
    row_offset = patch_height // 2
    col_offset = patch_width // 2
    #tmp_boundaries = np.tile(seg_boundaries[:,:,None]*255, (1,1,3))
    #cv2.imwrite('bd.png', tmp_boundaries)
    for i in range(total_bs_pixels):
        rand_idx = random.randint(0, total_bs_pixels-1)
        row_idx, col_idx = bd_index[0][rand_idx], bd_index[1][rand_idx]
        row_start, row_end = row_idx-row_offset, row_idx + col_offset+1
        col_start, col_end=  col_idx-col_offset, col_idx + col_offset +1

        count = 0
        label_patch = seg_label[row_start:row_end, col_start:col_end]
        if np.unique(label_patch).size == 2: #only consider the min patch with only two pixels
            patch_list.append(np.reshape(np.array([row_start, patch_height, col_start, patch_width]), (1, 4)))
            label_patch = select_label(label_patch)
            label_list.append(label_patch[None,:,:,:])
            #tmp_boundaries= cv2.rectangle(tmp_boundaries.astype(np.uint8), (col_start, row_start), (col_end, row_end), color=(0, 255, 0), thickness=1)

        if len(label_list) >= patch_num:
            break
    ''' 
    color_map = []
    for s in range(np.unique(seg_label).size):
       r = random.randint(0,255)
       g = random.randint(0,255)
       b = random.randint(0,255)
       color_map.extend([r,g,b])

    im = Image.fromarray(seg_label.astype(np.uint8))
    im.putpalette(color_map)
    im.save('label.png')
    cv2.imwrite('patch.png', tmp_boundaries)
    '''
    if len(label_list) == 0:
        patch_labels = np.ones((1, 4, patch_height,patch_width))
        patch_posi = np.reshape(np.array([1,patch_height,1,patch_width]), (1,4))
    else:
        patch_labels = np.concatenate(label_list, axis=0)
        patch_posi = np.concatenate(patch_list, axis=0)   

    return patch_labels, patch_posi

#============================================================================================
def collate_fn(data):
    im, label, patch_posi, patch_label = zip(*data)
    images = torch.stack(im,0)
    labels = torch.stack(label,0)

    return images, labels, patch_posi, patch_label
