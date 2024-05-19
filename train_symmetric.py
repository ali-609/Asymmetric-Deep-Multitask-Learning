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
from datasets import A2D2_steering,A2D2_seg,A2D2_symmetric,A2D2_depth,A2D2_box,a2d2_dataloader

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.first_hydra import HydraNet
from models.model_withUnet import HydraUNet
from models.resnet_bifpn_depth import ResNETBiFPN
import wandb
from datetime import datetime
import os
from losses import YOLOLoss,DepthLoss,MaskedMSELoss,MaskedL1Loss


import argparse

torch.manual_seed(42)
random.seed(42)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

parser = argparse.ArgumentParser()
parser.add_argument('--train-mode', choices=['segmentation', 'steering', 'box', 'depth', 'symmetric'], default='symmetric')

args = parser.parse_args()

train_mode = args.train_mode

print("Train mode: ",train_mode)

base_name = f"universal_{train_mode}"



A2D2_path_train,A2D2_path_val=a2d2_dataloader()



if train_mode == 'segmentation':
    A2D2_dataset_train = A2D2_seg(A2D2_path_train)
    A2D2_dataset_val = A2D2_seg(A2D2_path_val)
elif train_mode == 'steering':
    A2D2_dataset_train = A2D2_steering(A2D2_path_train)
    A2D2_dataset_val = A2D2_steering(A2D2_path_val)
elif train_mode == 'box':
    A2D2_dataset_train = A2D2_box(A2D2_path_train)
    A2D2_dataset_val = A2D2_box(A2D2_path_val)
elif train_mode == 'depth':
    A2D2_dataset_train = A2D2_depth(A2D2_path_train)
    A2D2_dataset_val = A2D2_depth(A2D2_path_val)
elif train_mode == 'symmetric':
    A2D2_dataset_train = A2D2_symmetric(A2D2_path_train)
    A2D2_dataset_val = A2D2_symmetric(A2D2_path_val)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('No of train samples', len(A2D2_dataset_train))
print('No of validation Samples', len(A2D2_dataset_val))




model = ResNETBiFPN()
model.to(device=device)

segmentation_loss = nn.CrossEntropyLoss()
steering_loss = nn.L1Loss()  
box_loss=YOLOLoss()
depth_loss=DepthLoss()


##############
##Parameters##
##############
BATCH_SIZE = 4

lr = 1e-5


n_epochs=40

grad_coef_str=1
grad_coef_seg=5
grad_coef_depth=8
grad_coef_box=8
##############
##Parameters##
##############

best_val_loss=999999


train_dataloader = DataLoader(A2D2_dataset_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
val_dataloader = DataLoader(A2D2_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

config = {
    "learning_rate": lr,
    "batch": 4
}



filename = f"weights/{base_name}_{current_time}.pth"
run_name = f"{base_name}_{current_time}"

wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)

segmentation_loss_value = 0
steering_loss_value = 0
depth_loss_value=0
boxes_loss_value=0

for epoch in range(n_epochs):
    
    model.train()

    total_training_loss = 0
    training_steering_loss = 0
    training_segmentation_loss = 0
    training_depth_loss = 0
    training_boxes_loss = 0




    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
  
        inputs = data["image"].to(device=device) 

        steering_output, segmentation_output, box_output, depth_output = model(inputs)
    

        if train_mode == 'segmentation':
            segmentation_label = data["A2D2_seg"].to(device=device) 

            segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)

            loss = segmentation_loss_value
        elif train_mode == 'steering':
            steering_label = data["A2D2_steering"].to(device=device) 

            steering_loss_value = steering_loss(steering_output, steering_label)

            loss = steering_loss_value
        elif train_mode == 'depth':
            depth_label = data["A2D2_depth"].to(device=device)  


            depth_loss_value = depth_loss(depth_output, depth_label)

            loss = depth_loss_value
        elif train_mode == 'box':
            boxes_label = data["A2D2_box"].to(device=device)

            boxes_loss_value = box_loss(box_output, boxes_label)

            loss = boxes_loss_value
        elif train_mode == 'symmetric':
            segmentation_label = data["A2D2_seg"].to(device=device) 
            steering_label = data["A2D2_steering"].to(device=device) 
            depth_label = data["A2D2_depth"].to(device=device)  
            boxes_label = data["A2D2_box"].to(device=device)  

            steering_loss_value = steering_loss(steering_output, steering_label)
            segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)
            depth_loss_value = depth_loss(depth_output, depth_label)
            boxes_loss_value = box_loss(box_output, boxes_label)  

            loss = segmentation_loss_value/grad_coef_seg + steering_loss_value/grad_coef_str + depth_loss_value/grad_coef_depth + boxes_loss_value/grad_coef_box

        
        #Backward
        loss.backward()
        optimizer.step() #Apply
        optimizer.zero_grad()
        
        
        #Logging
        training_segmentation_loss += segmentation_loss_value
        training_steering_loss += steering_loss_value
        total_training_loss += loss
        training_depth_loss += depth_loss_value
        training_boxes_loss += boxes_loss_value



        if train_mode == 'segmentation':
            wandb.log({"Training Segmentation Loss ": segmentation_loss_value})
        elif train_mode == 'steering':
            wandb.log({"Training Steering Loss ": steering_loss_value})
        elif train_mode == 'depth':
            wandb.log({"Training Depth Loss": depth_loss_value})
        elif train_mode == 'box':
            wandb.log({"Training Boxes Loss": boxes_loss_value})
        elif train_mode == 'symmetric':
            wandb.log({
        "Training Loss": loss,
        "Training Steering Loss ": steering_loss_value,
        "Training Segmentation Loss ": segmentation_loss_value,
        "Training Depth Loss": depth_loss_value,
        "Training Boxes Loss": boxes_loss_value,
    })
        

    avgTrainLoss = total_training_loss / len(train_dataloader)
    avgTrainSteeringLoss = training_steering_loss / len(train_dataloader)
    avgTrainSegmentationLoss = training_segmentation_loss / len(train_dataloader)
    avgTrainDepthLoss = training_depth_loss / len(train_dataloader)
    avgTrainBoxesLoss = training_boxes_loss / len(train_dataloader)

# #---------------------------------------------------------------------------------------------------------


    model.eval()
    total_validation_loss = 0
    validation_steering_loss = 0
    validation_segmentation_loss = 0
    validation_depth_loss = 0
    validation_boxes_loss = 0


    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            inputs = data["image"].to(device=device) 

            steering_output, segmentation_output, box_output, depth_output = model(inputs)



            if train_mode == 'segmentation':
                segmentation_label = data["A2D2_seg"].to(device=device) 
                segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)
                loss = segmentation_loss_value
            elif train_mode == 'steering':
                steering_label = data["A2D2_steering"].to(device=device) 
    
                steering_loss_value = steering_loss(steering_output, steering_label)
    
                loss = steering_loss_value
            elif train_mode == 'depth':
                depth_label = data["A2D2_depth"].to(device=device)  
    
                depth_loss_value = depth_loss(depth_output, depth_label)
    
                loss = depth_loss_value
            elif train_mode == 'box':
                boxes_label = data["A2D2_box"].to(device=device)
    
                boxes_loss_value = box_loss(box_output, boxes_label)
    
                loss = boxes_loss_value
            elif train_mode == 'symmetric':
                segmentation_label = data["A2D2_seg"].to(device=device) 
                steering_label = data["A2D2_steering"].to(device=device) 
                depth_label = data["A2D2_depth"].to(device=device)  
                boxes_label = data["A2D2_box"].to(device=device)  
    
                steering_loss_value = steering_loss(steering_output, steering_label)
                segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)
                depth_loss_value = depth_loss(depth_output, depth_label)
                boxes_loss_value = box_loss(box_output, boxes_label)  
    
                loss = segmentation_loss_value/grad_coef_seg + steering_loss_value/grad_coef_str + depth_loss_value/grad_coef_depth + boxes_loss_value/grad_coef_box
    
    
    
    
            validation_segmentation_loss += segmentation_loss_value
            validation_steering_loss += steering_loss_value
            validation_depth_loss += depth_loss_value
            validation_boxes_loss += boxes_loss_value
            total_validation_loss += loss


    avgValLoss = total_validation_loss / len(val_dataloader)
    avgValSteeringLoss = validation_steering_loss / len(val_dataloader)
    avgValSegmentationLoss = validation_segmentation_loss / len(val_dataloader)
    avgValDepthLoss = validation_depth_loss / len(val_dataloader)
    avgValBoxesLoss = validation_boxes_loss / len(val_dataloader)



    if train_mode == 'segmentation':
      wandb.log({
        "Average Train Segmentation Loss": avgTrainSegmentationLoss,
        "Average Validation Segmentation Loss": avgValSegmentationLoss,
    })
    elif train_mode == 'steering':
        wandb.log({
            "Average Train Steering Loss": avgTrainSteeringLoss,
            "Average Validation Steering Loss": avgValSteeringLoss,
        })
    elif train_mode == 'depth':
        wandb.log({
            "Average Train Depth Loss": avgTrainDepthLoss,
            "Average Validation Depth Loss": avgValDepthLoss,
        })
    elif train_mode == 'box':
        wandb.log({
            "Average Train Boxes Loss": avgTrainBoxesLoss,
            "Average Validation Boxes Loss": avgValBoxesLoss,
        })
    elif train_mode == 'symmetric':
        wandb.log({
            "Average Train Loss": avgTrainLoss,
            "Average Train Steering Loss": avgTrainSteeringLoss,
            "Average Train Segmentation Loss": avgTrainSegmentationLoss,
            "Average Train Depth Loss": avgTrainDepthLoss,
            "Average Train Boxes Loss": avgTrainBoxesLoss,
            "Average Validation Loss": avgValLoss,
            "Average Validation Steering Loss": avgValSteeringLoss,
            "Average Validation Segmentation Loss": avgValSegmentationLoss,
            "Average Validation Depth Loss": avgValDepthLoss,
            "Average Validation Boxes Loss": avgValBoxesLoss,
        })




    if avgValLoss<best_val_loss:
        best_val_loss=avgValLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)



