import sys
from pathlib import Path
from tqdm import tqdm
import datetime
import os
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
from datasets import A2D2_steering,A2D2_seg,A2D2_box

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from PIL import Image
# from models.first_hydra import HydraNet
# from models.pureUnet import HydraUNet
from models.model_withUnet import HydraUNet
from models.UNet import UNet
# from models.resnet_bifpn import ResNETBiFPN
from models.resnet_bifpn_depth import ResNETBiFPN

colors=np.array([
    # Car Colors
[255, 0, 0],  # Car 1
[200, 0, 0],  # Car 2
[150, 0, 0],  # Car 3
[128, 0, 0],  # Car 4

# Bicycle Colors
[182, 89, 6],  # Bicycle 1
[150, 50, 4],  # Bicycle 2
[90, 30, 1],  # Bicycle 3
[90, 30, 30],  # Bicycle 4

# Pedestrian Colors
[204, 153, 255],  # Pedestrian 1
[189, 73, 155],  # Pedestrian 2
[239, 89, 191],  # Pedestrian 3

# Truck Colors
[255, 128, 0],  # Truck 1
[200, 128, 0],  # Truck 2
[150, 128, 0],  # Truck 3

# Small Vehicles Colors
[0, 255, 0],  # Small vehicles 1
[0, 200, 0],  # Small vehicles 2
[0, 150, 0],  # Small vehicles 3

# Traffic Signal Colors
[0, 128, 255],  # Traffic signal 1
[30, 28, 158],  # Traffic signal 2
[60, 28, 100],  # Traffic signal 3

# Traffic Sign Colors
[0, 255, 255],  # Traffic sign 1
[30, 220, 220],  # Traffic sign 2
[60, 157, 199],  # Traffic sign 3

# Utility Vehicle Colors
[255, 255, 0],  # Utility vehicle 1
[255, 255, 200],  # Utility vehicle 2

# Other Colors
[233, 100, 0],  # Sidebars
[110, 110, 0],  # Speed bumper
[128, 128, 0],  # Curbstone
[255, 193, 37],  # Solid line
[64, 0, 64],  # Irrelevant signs
[185, 122, 87],  # Road blocks
[0, 0, 100],  # Tractor
[139, 99, 108],  # Non-drivable street
[210, 50, 115],  # Zebra crossing
[255, 0, 128],  # Obstacles / trash
[255, 246, 143],  # Poles
[150, 0, 150],  # RD restricted area
[204, 255, 153],  # Animals
[238, 162, 173],  # Grid structure
[33, 44, 177],  # Signal corpus
[180, 50, 180],  # Drivable cobblestone
[255, 70, 185],  # Electronic traffic
[238, 233, 191],  # Slow drive area
[147, 253, 194],  # Nature object
[150, 150, 200],  # Parking area
[180, 150, 200],  # Sidewalk
[72, 209, 204],  # Ego car
[200, 125, 210],  # Painted driv. instr.
[159, 121, 238],  # Traffic guide obj.
[128, 0, 255],  # Dashed line
[255, 0, 255],  # RD normal street
[135, 206, 255],  # Sky
[241, 230, 255],  # Buildings
[96, 69, 143],  # Blurred area
[53, 46, 82]  # Rain dirt

])


TRAIN_SPLIT = 0.8



# BASE_PATH = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/"
BASE_PATH="/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/" 


all_folders = sorted(os.listdir(BASE_PATH))
all_folders = all_folders[1:-3]
print(all_folders)


A2D2_path_train_seg=[]#A2D2_path_all[:int(len(A2D2_path_all) * TRAIN_SPLIT)]
A2D2_path_train_str=[]

A2D2_path_val_seg = []
A2D2_path_val_str = []  #A2D2_path_all[-int(len(A2D2_path_all) * VAL_SPLIT):]

A2D2_path_train  = [] 
A2D2_path_val = []

for folder in all_folders:
    folder_path = os.path.join(BASE_PATH, folder)
    
    # Get a list of all files in the current folder
    files_in_folder = sorted(glob.glob(os.path.join(folder_path, "camera/cam_front_center/*.png")))
    
    # Shuffle the list of files
    # random.shuffle(files_in_folder)
    
    # Calculate the split indices
    split_index = int(len(files_in_folder) * TRAIN_SPLIT)
    
    # Split the data into training and validation sets for the current folder
    train_set = files_in_folder[:split_index]
    val_set = files_in_folder[split_index:]
    
    # Accumulate the sets for each folder
    A2D2_path_train.extend(train_set)
    A2D2_path_val.extend(val_set)




A2D2_dataset_train_seg=A2D2_seg(A2D2_path_train)
A2D2_dataset_val_seg=A2D2_seg(A2D2_path_val)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




reverse_transform = transforms.Compose([
    transforms.ToPILImage(),

])


def process_array_multi(img):
    out = np.zeros((img.shape[1], img.shape[2],3), np.uint8)

    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            index=np.argmax(img[:,i,j])
            out[i,j,:]=colors[index]
        
    return out

def process_array_multi_reduced(img):
    out = np.zeros((img.shape[1], img.shape[2],3), np.uint8)

    # for i in range(img.shape[1]):
    #     for j in range(img.shape[2]):
    index=np.argmax(img,axis=0)

    index[index > 16] += 12


    out[:,:,:]=colors[index]
    print(out.shape)
        
    return out

def process_array_single(img):
    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    out = np.zeros((img.shape[1], img.shape[2],3), np.uint8)

    img_shape = (out.shape[0], out.shape[1], 3)
    img = np.zeros(img_shape, dtype=np.uint8)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j,:] = colors[out[i, j] // 4]  


    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        
    return out




sample_no=58

sample=A2D2_dataset_val_seg[sample_no]


truth=sample['A2D2_seg'] 


directory = f"results/segmentation/{sample_no}"

if not os.path.exists(directory):
    os.makedirs(directory)


ground_truth_file =   f"results/segmentation/{sample_no}/_truth.png"
real_image_file =   f"results/segmentation/{sample_no}/_real.png"
print("Truth: ")


real_image = sample['image']
output=process_array_multi_reduced(truth)
print(output.shape)
output_image=Image.fromarray(output, 'RGB')
real_image=reverse_transform(real_image)

# print(output_image)
# exit()

real_image.save(real_image_file)
output_image.save(ground_truth_file)


model=ResNETBiFPN()    
model.load_state_dict(torch.load('weights/universal_asymmetric_2024-05-07_20-56-31.pth' ))
model.eval()


print("Aymmetric: ")

sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

seg_output=  output[1][0]




output=process_array_multi_reduced(seg_output)
print(output.shape)
output_image=Image.fromarray(output, 'RGB')
output_file =   f"results/segmentation/{sample_no}/_asymmetric.png"
output_image.save(output_file)

print("Symmetric: ")

model=ResNETBiFPN()    
model.load_state_dict(torch.load('weights/universal_symmetric_2024-05-02_18-34-33.pth' ))
model.eval()

sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

seg_output=  output[1][0]




output=process_array_multi_reduced(seg_output)
print(output.shape)
output_image=Image.fromarray(output, 'RGB')
output_file =   f"results/segmentation/{sample_no}/_symmetric.png"
output_image.save(output_file)

print("Universal: ")

model=ResNETBiFPN()    
model.load_state_dict(torch.load('weights/universal_segmentation_2024-05-01_10-42-53.pth' ))
model.eval()

sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

seg_output=  output[1][0]




output=process_array_multi_reduced(seg_output)
print(output.shape)
output_image=Image.fromarray(output, 'RGB')
output_file =   f"results/segmentation/{sample_no}/_universal.png"
output_image.save(output_file)


print('Single Task: ')

model= UNet()
model.load_state_dict(torch.load('weights/segmentation_2024-04-21_00-16-07.pth' ))
model.eval()

sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

seg_output=  output[0]




output=process_array_multi_reduced(seg_output)
print(output.shape)
output_image=Image.fromarray(output, 'RGB')
output_file =   f"results/segmentation/{sample_no}/_single_task.png"
output_image.save(output_file)


