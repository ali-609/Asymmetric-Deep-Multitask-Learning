import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
from torchvision import transforms
# import torchvision.transforms.functional as F
import torchvision.models as models

import glob

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet50,resnet101,ResNet50_Weights
import cv2
from .unet_parts import *

class ResNETBiFPN(nn.Module):
    def __init__(self,bilinear=False):
        super(ResNETBiFPN, self).__init__()

        backbone_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,

        )

        self.backbone_1 = backbone_model.layer1

        self.backbone_2 = backbone_model.layer2

        self.backbone_3 = backbone_model.layer3

        self.backbone_4 = backbone_model.layer4
        
        self.channel_1 = DepthwiseConvBlock(256,256) 
        self.channel_2 = DepthwiseConvBlock(512,256)
        self.channel_3 = DepthwiseConvBlock(1024,256)
        self.channel_4 = DepthwiseConvBlock(2048,256)

        self.bifpn= BiFPNBlock(feature_size=256)
        self.bifpn_1= BiFPNBlock(feature_size=256)


        self.box_head = nn.Sequential(


                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(512, 512, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(512, 512, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),


                nn.Conv2d(512, 190, 3, padding=1),
                # nn.Sigmoid()
            )
        

        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # First decoder layer
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Second decoder layer 
            nn.LeakyReLU(),


            nn.ConvTranspose2d(128,64,kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 43, kernel_size=3, padding=1), 
        )

        self.depth_head= nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # First decoder layer
            nn.LeakyReLU(),


            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 1, 3, padding=1),  # Third decoder layer
            nn.ReLU()
        )
        
        self.steering_angle_backbone = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(16384, 100),
            nn.Linear(262144, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU()
        )
        self.steering_angle_head_21 = nn.Sequential(

            nn.Linear(10, 10)
        )


    def forward(self, x):

        p2= self.backbone(x)
        p3 = self.backbone_1(p2)
        p4 = self.backbone_2(p3)
        p5 = self.backbone_3(p4)
        p6 = self.backbone_4(p5)

        p3 = self.channel_1(p3)
        p4 = self.channel_2(p4)
        p5 = self.channel_3(p5)
        p6 = self.channel_4(p6)

        feature_map=self.bifpn([p3,p4,p5,p6])
        feature_map=self.bifpn_1([feature_map[0],feature_map[1],feature_map[2],feature_map[3]])
        # feature_map=self.bifpn_2([feature_map[0],feature_map[1],feature_map[2],feature_map[3]])

        segmentation_map =  self.segmentation_head(feature_map[1]) #1
        box_output=self.box_head(feature_map[2]) #2
        steering_angles_back = self.steering_angle_backbone(feature_map[3]) #3

        steering_angles_21=self.steering_angle_head_21(steering_angles_back)

        depth_output=self.depth_head(feature_map[0])

        
        # exit()

        
        return steering_angles_21, segmentation_map,box_output,depth_output
    

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.act = nn.LeakyReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)

        return self.act(x)
    

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        # self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        # self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 3))
        self.w1_relu = nn.LeakyReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 3))
        self.w2_relu = nn.LeakyReLU()
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x = inputs
        
        # Calculate Top-Down Pathway
        w1_i = self.w1_relu(self.w1)
        w1 = w1_i / (torch.sum(w1_i, dim=0) + self.epsilon)
        w2_i = self.w2_relu(self.w2)
        w2 = w2_i / (torch.sum(w2_i, dim=0) + self.epsilon)
        

        p6_td = p6_x                                      
        p5_td = self.p5_td(w1[0, 0] * p5_x + w1[1, 0] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 1] * p4_x + w1[1, 1] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 2] * p3_x + w1[1, 2] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td              
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        

        return [p3_out, p4_out, p5_out, p6_out]
    

    
