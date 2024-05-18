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


torch.manual_seed(42)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)


filename = f"weights/yolo_{current_time}.pth"


BATCH_SIZE = 4

TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.1


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

    A2D2_path_train_seg.extend(train_set[::2])
    A2D2_path_train_str.extend(train_set[1::2])

    A2D2_path_val_seg.extend(val_set[::2])
    A2D2_path_val_str.extend(val_set[1::2])



A2D2_dataset_train_seg=A2D2_seg(A2D2_path_train_seg)
A2D2_dataset_train_str=A2D2_steering(A2D2_path_train_str)

A2D2_dataset_val_seg=A2D2_seg(A2D2_path_val_seg)
A2D2_dataset_val_str=A2D2_steering(A2D2_path_val_str)

A2D2_dataset_train_box=A2D2_box(A2D2_path_train)
A2D2_dataset_val_box=A2D2_box(A2D2_path_val)

# A2D2_dataset_val=DatasetA2D2(A2D2_path_val)

# Kitty_dataset_train=DatasetKitty(Kitty_path_train)
# Kitty_dataset_val=DatasetKitty(Kitty_path_val)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = A2D2_dataset_train_box    #ConcatDataset([A2D2_dataset_train_seg,A2D2_dataset_train_str])      #AllDatasets2(A2D2_dataset_train,Kitty_dataset_train,BATCH_SIZE)
val_dataset = A2D2_dataset_val_box #  ConcatDataset([A2D2_dataset_val_seg,A2D2_dataset_val_str])      #AllDatasets2(A2D2_dataset_val,Kitty_dataset_val,BATCH_SIZE)


print('No of train samples', len(train_dataset))
print('No of validation Samples', len(val_dataset))




train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=24)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=24)


model = YOLOv1()
print(model)
# exit()
model.to(device=device)

box_loss=YOLOLoss()


lr = 1e-5
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

n_epochs=40

best_val_loss=999999

config = {
    "learning_rate": lr,

    "batch_backbone": BATCH_SIZE,

}

# torch.autograd.set_detect_anomaly(True)
run_name = f"box_{current_time}"
wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)
for epoch in range(n_epochs):

    model.train()

    training_box_loss = 0


    backbone_counter=0
    segmentation_counter=0
    steering_counter=0


    steering_loss_value=0
    batch_loss=0


    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
 
        

        inputs = data["image"].to(device=device) 
        box_label = data["A2D2_box"].to(device=device)


        box_output = model(inputs)


        box_loss_value = box_loss(box_output, box_label)



            
        box_loss_value.backward()





        optimizer.step()
        optimizer.zero_grad()


        
        

        training_box_loss += box_loss_value

        

        wandb.log({"Training Boxes Loss": box_loss_value})
        




    avgTrainSteeringLoss = training_box_loss / len(train_dataloader)


# #---------------------------------------------------------------------------------------------------------
# class custom()

    model.eval()
    total_validation_loss = 0
    validation_box_loss = 0
    validation_segmentation_loss = 0

    backbone_counter=0
    segmentation_counter=0
    steering_counter=0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):    
            
            inputs = data["image"].to(device=device) 
            steering_label = data["A2D2_box"].to(device=device)
    
    
            steering_output  = model(inputs)
    
            steering_loss_value = box_loss(steering_output, steering_label)

    


            validation_box_loss += steering_loss_value.item()



    avgValSteeringLoss = validation_box_loss / len(val_dataloader)



    wandb.log({

               "Average Train Boxes Loss": avgTrainSteeringLoss,

               "Average Validation Boxes Loss": avgValSteeringLoss,

    })


    if avgValSteeringLoss<best_val_loss:
        best_val_loss=avgValSteeringLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)

    



