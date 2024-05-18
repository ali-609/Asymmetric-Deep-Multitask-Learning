import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
# import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader,ConcatDataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models
# from skimage.util import random_noise
import glob
from datasets import A2D2_steering,A2D2_seg,A2D2_box
from losses import YOLOLoss 
from models.yolo_v1 import YOLOv1
# from models.resnet_bifpn import ResNETBiFPN
from models.resnet_bifpn_depth import ResNETBiFPN

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.steering_onlyUNet import HydraUNetSTR
# from UNet import UNet
import os
from datetime import datetime
import wandb
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
# from models.resnet_bifpn import ResNETBiFPN




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




A2D2_dataset_train_box=A2D2_box(A2D2_path_train)
A2D2_dataset_val_box=A2D2_box(A2D2_path_val)

# print(A2D2_dataset_train_box[0])

dataset=['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']





def box_visualization(image,boxes,name,threshold=1.0):
    image_np = np.array(TF.to_pil_image(image))


    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    # Iterate over each grid cell and draw bounding boxes
    for grid_x in range(boxes.shape[1]):
        for grid_y in range(boxes.shape[2]):
            for nth_anchor in range(10):
            # Confidence score
                confidence = boxes[nth_anchor*19, grid_x, grid_y]
                # If confidence score is above a threshold, draw the bounding box
                if confidence >= threshold:
                    if confidence>0.8929 and confidence<0.8931:
                        continue  # Adjust threshold as needed
                    print(confidence)
                    # Bounding box coordinates
                    x =  grid_x * (image_np.shape[1]/8)+ boxes[nth_anchor*19+1, grid_x, grid_y] * (image_np.shape[1]/8)  # x-coordinate within the grid
                    y =  grid_y * (image_np.shape[0]/8)+ boxes[nth_anchor*19+2, grid_x, grid_y] * (image_np.shape[0]/8)  # y-coordinate within the grid
                    width = boxes[nth_anchor*19+3, grid_x, grid_y] * (image_np.shape[1] /8) # Width of the box
                    height = boxes[nth_anchor*19+4, grid_x, grid_y] * (image_np.shape[0]/8)  # Height of the box
                    # Convert center coordinates to top-left coordinates
                    x1 = int(x - width / 2)
                    y1 = int(y - height / 2)
                    x2 = int(x + width / 2)
                    y2 = int(y + height / 2)
                    # Draw the bounding box
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                                           fill=False, edgecolor='r', linewidth=2))
               
                    # Get the class index
                    class_index = np.argmax(boxes[nth_anchor*19+5:nth_anchor*19+14, grid_x, grid_y])
                    class_name = dataset[class_index]
                    
                        # Show class label
                    # plt.text(x1, y1 - 5, class_name, color='r', fontsize=10, ha='left', va='center')
    
    plt.savefig(name)
    plt.close()

#844 6969 3131 196 144 1164 8114 1677 2274 97 139 
#519 533 829+ 995+ 1504+ 1663+ 1930+ 2014+ 2088+
##829
sample_no=829
output_threshold=0.9965


directory = f"results/boundingbox/{sample_no}"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

sample=A2D2_dataset_val_box[sample_no]

image=sample['image']
boxes_ground_truth=sample['A2D2_box']


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

### Truth
ground_truth_file =   f"results/boundingbox/{sample_no}/_truth.png"
print("Truth: ")
box_visualization(image, boxes_ground_truth, ground_truth_file,1.0)



###Single Task
model = YOLOv1()
model.load_state_dict(torch.load('weights/yolo_2024-05-03_19-36-16.pth' ))
model.eval()


sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


# boxes_prediction=output[2][0]
boxes_prediction=output[0]
print(boxes_prediction.shape)


prediction_file =   f"results/boundingbox/{sample_no}/_single_task.png"
print("Single task: ")
box_visualization(image, boxes_prediction, prediction_file,output_threshold)

### Asymmetric
model=ResNETBiFPN()    
model.load_state_dict(torch.load('weights/universal_asymmetric_2024-05-07_20-56-31.pth' ))
# model.to(device=device)
sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


boxes_prediction=output[2][0]
# boxes_prediction=output[0]
print(boxes_prediction.shape)


prediction_file =   f"results/boundingbox/{sample_no}/_asymmetric.png"
print("Asymmetric task: ")
box_visualization(image, boxes_prediction, prediction_file,output_threshold)


###
model=ResNETBiFPN()    
model.load_state_dict(torch.load('weights/universal_symmetric_2024-05-02_18-34-33.pth' ))
# model.to(device=device)
sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


boxes_prediction=output[2][0]
# boxes_prediction=output[0]
print(boxes_prediction.shape)


prediction_file =   f"results/boundingbox/{sample_no}/_symmetric.png"
print("Symmetric task: ")
box_visualization(image, boxes_prediction, prediction_file,output_threshold)

model=ResNETBiFPN()    
model.load_state_dict(torch.load('weights/universal_box_2024-05-02_18-42-46.pth' ))
# model.to(device=device)
sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


boxes_prediction=output[2][0]
# boxes_prediction=output[0]
print(boxes_prediction.shape)


prediction_file =   f"results/boundingbox/{sample_no}/_universal.png"
print("Universal: ")
box_visualization(image, boxes_prediction, prediction_file,output_threshold)