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
from datasets import *
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
import json
A2D2_path=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))
Kitty_path=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/Kitty/semantics/training/image_2/*png"))

A2D2_dataset=DatasetA2D2(A2D2_path)
Kitty_data=DatasetKitty(Kitty_path)

test_data=AllDatasets(A2D2_dataset,Kitty_data)
print(test_data[len(A2D2_dataset)+3])

Record the start time
start_time = time.time()

for i in range(32):
    print(test_data[i])


end_time = time.time()

# # Calculate the time spent
time_spent = end_time - start_time

print(f"Time spent on the process: {time_spent:.2f} seconds")

dirs=sorted(glob.glob('/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/bus/*/bus/*camera_frontcenter*.json'))


print(len(dirs))

A2D2_path=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))

print(len(A2D2_path))


print(len(A2D2_path)-len(dirs))

# for file in dirs:
    # json_file = open(file) 
    # json_data = json.load(json_file)
    # angles=json_data.get('steering_angles')
    # if angles==[]:
    #   print(file)
      