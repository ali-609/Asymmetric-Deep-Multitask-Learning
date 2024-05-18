import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models
from skimage.util import random_noise
import glob
from datasets import DatasetA2D2

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from .unet_parts import *

class HydraUNetSTR(nn.Module):
    def __init__(self,bilinear=False):
        super(HydraUNetSTR, self).__init__()
        
        # self.feature_extractor = nn.Sequential(
        # DoubleConv(3, 64),
        # Down(64, 128),
        # Down(128, 256),
        # Down(256, 512),
        # Down(512, 512)
        # )

        
        # self.segmentation_predictor = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        #     DoubleConv(512, 512),
        #     nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        #     DoubleConv(256, 256),
        #     nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        #     DoubleConv(128, 128),
        #     nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        #     DoubleConv(64, 64),
        #     nn.Conv2d(64, 3, kernel_size=1)
        # )


        self.inc = (DoubleConv(3, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 43)) #55
        # self.soft_max = nn.Softmax(dim=1)
        
        self.steering_angle_backbone = nn.Sequential(
            #CNN for getting necessary information 
            nn.Flatten(),
            nn.Linear(200704, 100),
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            # nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU()
        )
        self.steering_angle_head_21 = nn.Sequential(

            nn.Linear(10, 10)
        )
        self.steering_angle_head_1 = nn.Sequential(

            nn.Linear(10, 1)
        )

    def forward(self, x):
        # Extract features from the input image
        # features = self.feature_extractor(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        steering_angles_back = self.steering_angle_backbone(x5)
        steering_angles_21=self.steering_angle_head_21(steering_angles_back)
        # softmax_segmentation_map = self.soft_max(segmentation_map)
        
        
        # Predict segmentation
        #segmentation_map = self.segmentation_predictor(features)
        
        return steering_angles_21