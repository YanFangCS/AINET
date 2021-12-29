import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *

def compute_semantic_pos_loss(prob_in, labxy_feat, pix_emb, patch_posi, patch_label, alpha, pos_weight = 0.003,  kernel_size=16, curr_epoch=0, epoch_stone=3000):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    if curr_epoch >= epoch_stone: #start finetune
        bdl = boundary_perceiving_loss(pix_emb, patch_posi, patch_label)
        # empirically we find timing 0.005 tend to better performance
        loss_sum =  0.005 * (loss_sem + loss_pos + alpha * bdl)
    else:
        loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

#======================================================================================
def boundary_perceiving_loss(feat_map, patch_posi, patch_label):
    bs, c, h, w = feat_map.shape
    device = feat_map.device
    patch_loss = torch.tensor([0.]).to(device)
    for i in range(bs):
        label = patch_label[i].to(device)
        patches = patch_posi[i]
        feat = feat_map[i]
        
        patch_num = patches.shape[0]
        patches_i = []
        labels_i = []
        for k in range(patch_num):
            patch = patches[k]
            patch_label_i = label[k]
            feat_patch = torch.narrow(feat, 1, patch[0], patch[1])
            feat_patch = torch.narrow(feat_patch, 2, patch[2], patch[3])
            patches_i.append(feat_patch)
            labels_i.append(patch_label_i)

        patch_stack = torch.stack(patches_i, dim=0)            
        label_stack = torch.stack(labels_i, dim=0)            
        patch_loss_i = patch_classify(patch_stack, label_stack)
        patch_loss += patch_loss_i

    return patch_loss / bs

def patch_classify(feat, label):
    def simi_func(anchor_emb, emb):
        norm = torch.sum(torch.abs(anchor_emb-emb), dim=-1)
        simi = 2.0 / (1 + torch.exp(norm).clamp(min=1e-8, max=1e15))

        return simi
    #feat: c x h x w
    #label: 4 x h x w
    patch_num, c,h,w = feat.shape
    label_num = torch.sum(torch.sum(label,dim=-1),dim=-1)
    feat1_1 = feat * label[:,0:1]
    feat1_1 = torch.sum(torch.sum(feat1_1,dim=-1),dim=-1) / (label_num[:,0:1]+1)

    feat1_2 = feat * label[:,1:2]
    feat1_2 = torch.sum(torch.sum(feat1_2,dim=-1),dim=-1) / (label_num[:,1:2]+1)

    feat2_1 = feat * label[:,2:3]
    feat2_1 = torch.sum(torch.sum(feat2_1,dim=-1),dim=-1) / (label_num[:,2:3]+1)

    feat2_2 = feat * label[:,3:4]
    feat2_2 = torch.sum(torch.sum(feat2_2,dim=-1),dim=-1) / (label_num[:,3:4]+1)

    same_simi1 = simi_func(feat1_1, feat1_2)
    same_simi2 = simi_func(feat2_1, feat2_2)
    
    cross_simi1 = simi_func(feat1_1, feat2_1)
    cross_simi2 = simi_func(feat1_2, feat2_2)

    same_loss = -(torch.log(same_simi1+1e-8) + torch.log(same_simi2+1e-8)) / 2.
    cross_loss = -(torch.log(1-cross_simi1 + 1e-8) + torch.log(1-cross_simi2 + 1e-8))/2.

    return torch.mean(same_loss) + torch.mean(cross_loss)
