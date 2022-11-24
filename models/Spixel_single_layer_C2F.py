import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *
import pdb
import math

# define the function includes in import *
__all__ = [
    'SpixelNet1l','SpixelNet1l_bn'
]

class Recurrent_Attn(nn.Module):
    def __init__(self, num_class):
        super(Recurrent_Attn, self).__init__()

        self.QConv1 = conv(True, 256, 256, 3)
        self.KConv1 = conv(True, 256, 256, 3)
        self.VConv1 = conv(True, 256, 256, 3)

        kernel_size = 3
        stride = 1
        self.num_class = num_class
        self.QConv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.KConv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        self.VConv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

    def self_attn(self, Q, K, V):
        #projection the Q, K and V 
        b, c, h, w = Q.shape
        #B x C x H x W ==> B X C x 9 x H x W
        K_unfold = F.unfold(K, (3,3), padding=1)
        K_unfold = K_unfold.view(b, c, 9, h, w)

        #B x C x H x W ==> B X C x 9 x H x W
        V_unfold = F.unfold(V, (3,3), padding=1)
        V_unfold = V_unfold.view(b, c, 9, h, w)

        Q_unfold = Q.unsqueeze(2)

        #dot = torch.exp(Q_unfold * K_unfold / math.sqrt(c*1.))
        #dot = torch.sum(Q_unfold * K_unfold, dim=1, keepdim=True) / math.sqrt(c*1.)
        dot = Q_unfold * K_unfold / math.sqrt(c*1.)
        dot = F.softmax(dot, dim=2)

        attn = torch.sum(dot * V_unfold, dim=2)
        
        return attn

    def forward(self, x):
        Q1 = self.QConv1(x)
        K1 = self.KConv1(x)
        V1 = self.VConv1(x)

        attn1 = self.self_attn(Q1, K1, V1)
        attn1 = attn1 + x

        Q2 = self.QConv2(attn1)
        K2 = self.KConv2(attn1)
        V2 = self.VConv2(attn1)

        attn2 = self.self_attn(Q2, K2, V2)

        return attn2 


class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self,dataset='', batchNorm=True):
        super(SpixelNet,self).__init__()
        if dataset == 'ISIC_2017' or dataset=='BDS500':
            input_chs = 3
        elif dataset == 'BraTS2017':
            input_chs = 4
        elif dataset == 'ACDC' or dataset == 'TCIA':
            input_chs = 1
        #input_chs = 4
        self.batchNorm = batchNorm
        self.assign_ch = 9
        self.Train = False

        self.conv0a = conv(self.batchNorm, input_chs, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.conv5a = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2)
        self.conv5b = conv(self.batchNorm, 512, 512, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
        #===============================================
        mask_select0 = torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).reshape(3,3)
        self.mask_select0 = mask_select0.cuda()


        self.sp_pred = Recurrent_Attn(num_class=50)
        self.bridge_sp1 = conv(self.batchNorm, 256, 64, kernel_size=3)
        self.bridge_sp2 = conv(self.batchNorm, 64, 16, kernel_size=3)

        self._bridge_sp1 = conv(self.batchNorm, 512, 128, kernel_size=3)
        self._bridge_sp2 = conv(self.batchNorm, 128, 32, kernel_size=3)

        self.merge_sp = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        self._merge_sp = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

    def expand_sp(self, sp_feat, level=0):
        sp_feat = F.pad(sp_feat, [1,1,1,1])
        b, c, h, w = sp_feat.shape
        output_list = []
        interval = 16 if level ==0 else 16
        for i in range(1,h-1):
            row_list = []
            for j in range(1,w-1):
                sp_patch = sp_feat[:,:, (i-1):(i+2), (j-1):(j+2)]
                sp_patch = sp_patch.repeat(1,1,interval,interval)
                row_list.append(sp_patch)

            output_list.append(torch.cat(row_list, dim=-1))
        
        output = torch.cat(output_list, dim=-2)
        #output = output * (1-self.mask_select)

        return output

    def expand_pixel(self, pixel_feat, level=0):
        b,c,h,w = pixel_feat.shape
        pixel_feat = pixel_feat.view(b,c,h,1,w,1)
        pixel_feat = pixel_feat.repeat(1,1,1,3,1,3)
        pixel_feat = pixel_feat.reshape(b,c,h*3,w*3)
       
        if level == 0:
            pixel_feat = pixel_feat * self.mask_select
        else:
            pixel_feat = pixel_feat * self.mask_select_
    
        return pixel_feat

    def forward(self, x):
        b,c,h,w = x.shape
        mask_select = self.mask_select0.repeat(h,w)
        self.mask_select = mask_select.view(1,1, h*3, w*3).float()

        mask_select = self.mask_select0.repeat(h//2,w//2)
        self.mask_select_ = mask_select.view(1,1, h//2*3, w//2*3).float()

        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95

        out5_attn = self.sp_pred(out5)
        out5 = out5 + out5_attn
        out6 = self.conv5b(self.conv5a(out5)) #95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        sp_map = self._bridge_sp2(self._bridge_sp1(out6))
        sp_expand = self.expand_sp(sp_map, level=1)
        pixel_expand = self.expand_pixel(out_conv1_1, level=1)
        if sp_expand.shape[-1] != pixel_expand.shape[-1] or sp_expand.shape[-2] != pixel_expand.shape[-2]:
            sp_expand = F.interpolate(sp_expand, (pixel_expand.shape[-2], pixel_expand.shape[-1]), mode='bilinear', align_corners=True)
        merged = sp_expand + pixel_expand
        out_ = self._merge_sp(merged)
        out_conv1_1 = out_conv1_1 + out_ 

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        
        sp_map = self.bridge_sp2(self.bridge_sp1(out5))
        sp_expand = self.expand_sp(sp_map)
        pixel_expand = self.expand_pixel(out_conv0_1)
        merged = sp_expand + pixel_expand
        #merged = torch.cat([pixel_expand, sp_expand], dim=1)
        out = self.merge_sp(merged)
        out = out + out_conv0_1

        mask0 = self.pred_mask0(out)
        prob0 = self.softmax(mask0)

        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def SpixelNet1l( data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(dataset='BDS500',data=None):
    # model with batch normalization
    model = SpixelNet(dataset=dataset,batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
#
