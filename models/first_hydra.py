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

class HydraNet(nn.Module):
    def __init__(self):
        super(HydraNet, self).__init__()
        self.feature_extractor = nn.Sequential( #UNet
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization after the first convolution
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Batch normalization after the second convolution
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Batch normalization after the third convolution
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.segmentation_predictor = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # First decoder layer
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Second decoder layer
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Third decoder layer
        )

        self.steering_angle_backbone = nn.Sequential(
            nn.Linear(200704, 256),  # Adjust the input size to match the feature size
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()

        )
        self.steering_angle_head_21 = nn.Sequential(

            nn.Linear(128, 21)
        )
        self.steering_angle_head_1 = nn.Sequential(

            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Extract features from the input image
        features = self.feature_extractor(x)
        
        # Determine the size for the steering angle prediction
        steering_input_size = features.size(0), -1
        # Flatten the features for steering angle prediction
        steering_input = features.view(steering_input_size)
        # Predict steering angles
        steering_angles_back = self.steering_angle_backbone(steering_input)
        steering_angles_21=self.steering_angle_head_21(steering_angles_back)
        # Predict segmentation
        segmentation_map = self.segmentation_predictor(features)
        
        return steering_angles_21, segmentation_map