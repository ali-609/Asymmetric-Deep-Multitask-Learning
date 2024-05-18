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

class HydraUNet(nn.Module):
    def __init__(self,bilinear=False):
        super(HydraUNet, self).__init__()
        


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

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)


        
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


    def forward(self, x):

        x1 = self.inc(x)
        # x1 = self.dropout1(x1)
        x2 = self.down1(x1)
        # x2 = self.dropout2(x2)
        x3 = self.down2(x2)
        # x3 = self.dropout3(x3)
        x4 = self.down3(x3)
        # x4 = self.dropout4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        segmentation_map = self.outc(x)

        steering_angles_back = self.steering_angle_backbone(x5)
        steering_angles_21=self.steering_angle_head_21(steering_angles_back)

        
        return steering_angles_21, segmentation_map