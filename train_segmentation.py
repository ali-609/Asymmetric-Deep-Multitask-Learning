import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
# import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import DataLoader



import glob
from datasets import A2D2_seg


import random
import torch
import torch.nn as nn


import cv2
from models.UNet import UNet

import os
from datetime import datetime
import wandb


torch.manual_seed(42)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

# Construct the filename with the current time
filename = f"weights/segmentation_{current_time}.pth"


BATCH_SIZE = 16

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2



BASE_PATH_BOX = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/" #test for boxes



all_folders_box = sorted(os.listdir(BASE_PATH_BOX))
all_folders_box = all_folders_box[1:-3]



A2D2_path_train=[]


A2D2_path_val = []



for folder in all_folders_box:
    folder_path = os.path.join(BASE_PATH_BOX, folder)
    
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

train_dataset = A2D2_dataset_train_seg#ConcatDataset([A2D2_dataset_train_seg,A2D2_dataset_train_str,A2D2_dataset_train_box,A2D2_dataset_train_dep])      
val_dataset =   A2D2_dataset_val_seg#ConcatDataset([A2D2_dataset_val_seg,A2D2_dataset_val_str,A2D2_dataset_val_box,A2D2_dataset_val_dep])      

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=24)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=24)


model = UNet()
model.to(device=device)
segmentation_loss =  nn.CrossEntropyLoss()  #1 




lr = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_epochs=40

best_val_loss=999999
config = {
    "learning_rate": lr,
    "batch_segmentation": BATCH_SIZE
}


run_name = f"segmentation_{current_time}"
wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)

for epoch in range(n_epochs):
    
    model.train()

    total_training_loss = 0
    training_steering_loss = 0
    training_segmentation_loss = 0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
        # model.train()        
        
        inputs = data["image"].to(device=device) 
        segmentation_label = data["A2D2_seg"].to(device=device)


        

        #Output
        optimizer.zero_grad()
        segmentation_output = model(inputs)


        # Loss calculation
        segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)

        #Backward
        segmentation_loss_value.backward()
        optimizer.step()
        
        #Logging
        
        training_segmentation_loss += segmentation_loss_value


        wandb.log({"Training Segmentation Loss ": segmentation_loss_value})
        

    avgTrainLoss = total_training_loss / len(A2D2_dataset_train_seg)
    avgTrainSegmentationLoss = training_segmentation_loss / len(train_dataloader)

# #---------------------------------------------------------------------------------------------------------


    model.eval()
    total_validation_loss = 0

    validation_segmentation_loss = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            inputs = data["image"].to(device=device) 
            segmentation_label = data["A2D2_seg"].to(device=device)

            segmentation_output = model(inputs)

            segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)

            
            validation_segmentation_loss += segmentation_loss_value.item()


    avgValLoss = total_validation_loss / len(val_dataloader)
    avgValSegmentationLoss = validation_segmentation_loss / len(val_dataloader)


    wandb.log({
               "Average Train Segmentation Loss": avgTrainSegmentationLoss,
               "Average Validation Segmentation Loss": avgValSegmentationLoss
    })



    if avgValSegmentationLoss<best_val_loss:
        best_val_loss=avgValSegmentationLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)

    


