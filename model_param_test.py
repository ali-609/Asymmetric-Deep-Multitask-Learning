import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader,ConcatDataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models
from skimage.util import random_noise
import glob
from datasets import A2D2_steering,A2D2_seg,A2D2_symmetric,A2D2_depth,A2D2_box

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
from models.pilotnet import PilotNet
import cv2
from models.first_hydra import HydraNet
from models.model_withUnet import HydraUNet
from models.resnet_bifpn_depth import ResNETBiFPN
import wandb
from datetime import datetime
import os
from losses import YOLOLoss,DepthLoss,MaskedMSELoss,MaskedL1Loss
from models.DenseDepth import PTModel
from models.UNet import UNet
from models.yolo_v1 import YOLOv1


model1=PilotNet()

sum1=sum(p.numel() for p in model1.parameters()if p.requires_grad)
print("PilotNet: ",sum1)

model2=PTModel()

sum2=sum(p.numel() for p in model2.parameters()if p.requires_grad)
print("DenseDepth: ",sum2)

model3=UNet()

sum3=sum(p.numel() for p in model3.parameters()if p.requires_grad)
print("UNet: ",sum3)

model4=YOLOv1()

sum4=sum(p.numel() for p in model4.parameters()if p.requires_grad)
print("YOLO V1: ",sum4)

agg_sum=sum1+sum2+sum3+sum4
print("Aggeregation of models:",agg_sum)

model=ResNETBiFPN()


sum_mtl=sum(p.numel() for p in model.parameters()if p.requires_grad)
print("MTL: ",sum_mtl)

