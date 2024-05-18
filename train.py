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

import glob
from datasets import A2D2_steering,A2D2_seg,A2D2_box,A2D2_depth

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.first_hydra import HydraNet
from models.model_withUnet import HydraUNet
# from models.dino_backbone import DinoBackBone
from models.resnet_bifpn import ResNETBiFPN
from losses import YOLOLoss 
import wandb
from datetime import datetime
import os
torch.manual_seed(42)
random.seed(42)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

base_name = "asymmetric_three_task"
# Construct the filename with the current time

# torch.backends.cuda.matmul.allow_tf32 = False  # Can help reduce memory usage
# torch.backends.cudnn.deterministic = True  # Ensures reproducibility
# torch.cuda.set_per_process_memory_fraction(1.0, device=None)
# torch.cuda.set_per_process_memory_growth(True)
# Set max_split_size_mb to avoid fragmentation
# torch.cuda.set_per_process_memory_fraction(0.5, device=None)
# torch.cuda.set_per_process_memory_growth(True)

# Save the model with the constructed filename


def path_fixer(path, batch):
    if len(path) % batch == 0:
        return path

    n_samples = batch - (len(path) % batch)
    samples = random.sample(path, n_samples)

    for sample in samples:
        path.append(sample)

    return path

BATCH_SIZE = 1

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2


# random.shuffle(A2D2_path_all)
# BASE_PATH = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/"
BASE_PATH_BOX = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/" #test for boxes


# all_folders = sorted(os.listdir(BASE_PATH))
# all_folders = all_folders[1:-3]
all_folders_box = sorted(os.listdir(BASE_PATH_BOX))
all_folders_box = all_folders_box[0:-2]

# print(all_folders_box)
# exit()

A2D2_path_train_seg=[]
A2D2_path_train_str=[]
A2D2_path_train=[]

A2D2_path_val_seg = []
A2D2_path_val_str = [] 
A2D2_path_val = []

# A2D2_path_train  = [] 
# A2D2_path_val = []

# for folder in all_folders:
#     folder_path = os.path.join(BASE_PATH, folder)
    
#     # Get a list of all files in the current folder
#     files_in_folder = sorted(glob.glob(os.path.join(folder_path, "camera/cam_front_center/*.png")))
    
#     # Shuffle the list of files
#     # random.shuffle(files_in_folder)
    
#     # Calculate the split indices
#     split_index = int(len(files_in_folder) * TRAIN_SPLIT)
    
#     # Split the data into training and validation sets for the current folder
#     train_set = files_in_folder[:split_index]
#     val_set = files_in_folder[split_index:]
    


#     A2D2_path_train_seg.extend(train_set[::2])
#     A2D2_path_train_str.extend(train_set[1::2])

#     A2D2_path_val_seg.extend(val_set[::2])
#     A2D2_path_val_str.extend(val_set[1::2])



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
A2D2_dataset_train_str=A2D2_steering(A2D2_path_train)
A2D2_dataset_train_box=A2D2_box(A2D2_path_train)
A2D2_dataset_train_dep=A2D2_depth(A2D2_path_train)

A2D2_dataset_val_seg=A2D2_seg(A2D2_path_val)
A2D2_dataset_val_str=A2D2_steering(A2D2_path_val)
A2D2_dataset_val_box=A2D2_box(A2D2_path_val)
A2D2_dataset_val_dep=A2D2_depth(A2D2_path_val)


# A2D2_dataset_train_seg=A2D2_seg(A2D2_path_train_seg)
# A2D2_dataset_train_str=A2D2_steering(A2D2_path_train_str)
# A2D2_dataset_train_box=A2D2_box(A2D2_path_train_box)

# A2D2_dataset_val_seg=A2D2_seg(A2D2_path_val_seg)
# A2D2_dataset_val_str=A2D2_steering(A2D2_path_val_str)
# A2D2_dataset_val_box=A2D2_box(A2D2_path_val_box)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ConcatDataset([A2D2_dataset_train_seg,A2D2_dataset_train_str,A2D2_dataset_train_box,A2D2_dataset_train_dep])      
val_dataset =   ConcatDataset([A2D2_dataset_val_seg,A2D2_dataset_val_str,A2D2_dataset_val_box,A2D2_dataset_val_dep])      


print('No of total train samples', len(train_dataset))
print('No of total validation Samples', len(val_dataset),'\n')

print('No of segmentation train samples', len(A2D2_dataset_train_seg))
print('No of segmentation validation Samples', len(A2D2_dataset_val_seg),'\n')


print('No of steering train samples', len(A2D2_dataset_train_str))
print('No of steering validation Samples', len(A2D2_dataset_val_str),'\n')

print('No of box train samples', len(A2D2_dataset_train_box))
print('No of box validation Samples', len(A2D2_dataset_val_box),'\n')





train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=16)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16)


model = ResNETBiFPN()
# model.load_state_dict(torch.load('weights/asymmetric_three_task_2024-02-29_02-15-10.pth'))
model.to(device=device)

# # Loss functions
segmentation_loss = nn.CrossEntropyLoss()
steering_loss = nn.L1Loss()  
box_loss=YOLOLoss()



backbone_params    =list(model.backbone.parameters())+list(model.backbone_1.parameters())+list(model.backbone_2.parameters())+list(model.backbone_3.parameters())+list(model.backbone_4.parameters())+list(model.channel_1.parameters())+list(model.channel_2.parameters())+list(model.channel_3.parameters())+list(model.channel_4.parameters())+list(model.bifpn.parameters())+list(model.bifpn_1.parameters())
segmentation_params=list(model.segmentation_head.parameters())
steering_params=list(model.steering_angle_backbone.parameters())+list(model.steering_angle_head_21.parameters())
box_params=list(model.box_head.parameters())

##############
##Parameters##
##############
batch_backbone=10
batch_steering=6
batch_segmentation=6
batch_box=8


backbone_lr = 1e-3
segmentation_lr= 3e-3
steering_lr = 2e-5
box_lr= 1e-5

##############
##Parameters##
##############


backbone_optimizer = torch.optim.Adam(backbone_params, lr=backbone_lr/batch_backbone)
segmentation_optimizer = torch.optim.Adam(segmentation_params, lr=segmentation_lr)
steering_optimizer = torch.optim.Adam(steering_params, lr=steering_lr)
box_optimizer = torch.optim.Adam(box_params, lr=box_lr)

#Check relation between batch size,loss coef and learning rate
# exit()
n_epochs=100

best_val_loss=999999

config = {
    "learning_rate": backbone_lr,
    "segmentation_learning_rate": segmentation_lr,
    "steering_learning_rate": steering_lr,
    "box_learning_rate": box_lr,
    "segmentation_loss": segmentation_loss.__class__.__name__,
    "steering_loss": steering_loss.__class__.__name__,
    "batch_backbone": batch_backbone,
    "batch_steering": batch_steering,
    "batch_box": batch_box,
    "batch_segmentation":batch_segmentation
}

# torch.autograd.set_detect_anomaly(True)

filename = f"weights/{base_name}_{current_time}.pth"
run_name = f"{base_name}_{current_time}"

wandb.init(project="test", config=config, name=run_name)
wandb.watch(model)
for epoch in range(n_epochs):

    model.train()

    total_training_loss = 0
    training_steering_loss = 0
    training_segmentation_loss = 0
    training_box_loss = 0 #Done
    

    backbone_counter=0
    segmentation_counter=0
    steering_counter=0
    box_counter=0 #Done

    total_loss=0
    steering_loss_value=0
    segmentation_loss_value=0
    box_loss_value=0

    batch_loss=0
    batch_segmentation_loss=0
    batch_steering_loss=0
    batch_box_loss=0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):

        if data["dt_label"][0]=='A2D2_steering':
            # print('hero')
            steering_counter=steering_counter+1
            backbone_counter=backbone_counter+1
            inputs = data["image"].to(device=device) 
            steering_label = data["steering"].to(device=device)


            steering_output ,segmentation_output,box_output = model(inputs)
            # x= model(inputs)
            # print(segmentation_output.shape)
            # exit()


            steering_loss_value = steering_loss(steering_output, steering_label)
            #metrics
            training_steering_loss += steering_loss_value
            total_training_loss += steering_loss_value
            
            batch_steering_loss+=steering_loss_value
            batch_loss+=steering_loss_value
            # print(steering_output)


            steering_loss_value = steering_loss_value/batch_steering
            steering_loss_value.backward()
            # print('backward done')


            if steering_counter%batch_steering==0:
                steering_optimizer.step()
                steering_optimizer.zero_grad()
                steering_counter=0
                
                wandb.log({"Training Steering Loss ": batch_steering_loss/batch_steering})

                batch_steering_loss=0
        
        if data["dt_label"][0]  == 'A2D2_box':
            # continue
            box_counter=box_counter+1
            backbone_counter=backbone_counter+1
            
            inputs = data["image"].to(device=device)
            box_label=data["box"].to(device=device)

            steering_output ,segmentation_output,box_output = model(inputs)
            # print(box_label.shape)
            # print(box_output.shape)
            # exit()
            
            box_loss_value = box_loss(box_output, box_label)
            training_box_loss += box_loss_value
            total_training_loss += box_loss_value


            batch_box_loss += box_loss_value
            batch_loss += box_loss_value
            box_loss_value = box_loss_value/batch_box/10


            box_loss_value.backward()



            if box_counter%batch_box==0:
                box_optimizer.step()
                box_optimizer.zero_grad()
                box_counter=0
                
                # recallate loss and log

                wandb.log({"Training Bounding Box Loss ": batch_box_loss/batch_box})
                batch_box_loss=0
                
        if data["dt_label"][0]=='A2D2_seg':
            segmentation_counter=segmentation_counter+1
            backbone_counter=backbone_counter+1

            inputs = data["image"].to(device=device) 
            segmentation_label = data["segmentation"].to(device=device)

            steering_output ,segmentation_output,box_output = model(inputs)


            segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)
            training_segmentation_loss += segmentation_loss_value
            total_training_loss += segmentation_loss_value


            batch_segmentation_loss += segmentation_loss_value
            batch_loss += segmentation_loss_value

            segmentation_loss_value = segmentation_loss_value/batch_segmentation/5


            segmentation_loss_value.backward()



            if segmentation_counter%batch_segmentation==0:
                segmentation_optimizer.step()
                segmentation_optimizer.zero_grad()
                segmentation_counter=0
                

                wandb.log({"Training Segmentation Loss ": batch_segmentation_loss/batch_segmentation})
                batch_segmentation_loss=0
        

        if backbone_counter%batch_backbone==0:
            backbone_optimizer.step()
            backbone_optimizer.zero_grad()
            backbone_counter=0
            total_loss=segmentation_loss_value+steering_loss_value+box_loss_value
            # total_training_loss += total_loss
            wandb.log({"Training Loss ": batch_loss/batch_backbone})
            batch_loss=0


        

        



    avgTrainLoss = total_training_loss / len(train_dataloader)
    avgTrainSteeringLoss = training_steering_loss / len(A2D2_dataset_train_str)
    avgTrainSegmentationLoss = training_segmentation_loss / len(A2D2_dataset_train_seg)
    avgTrainBoxLoss = training_box_loss / len(A2D2_dataset_train_box)

# #---------------------------------------------------------------------------------------------------------
# class custom()

    model.eval()
    total_validation_loss = 0
    validation_steering_loss = 0
    validation_segmentation_loss = 0
    validation_box_loss = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):        

            if data["dt_label"][0]=='A2D2_steering':
                inputs = data["image"].to(device=device) 
                steering_label = data["steering"].to(device=device)


                steering_output ,segmentation_output,box_output = model(inputs)
    
                steering_loss_value = steering_loss(steering_output, steering_label)
                validation_steering_loss += steering_loss_value.item()
                loss = steering_loss_value 

            

                
            if data["dt_label"][0]=='A2D2_seg':
                inputs = data["image"].to(device=device) 
                segmentation_label = data["segmentation"].to(device=device)
    
                steering_output ,segmentation_output,box_output = model(inputs)
    
                
    
                segmentation_loss_value = segmentation_loss(segmentation_output, segmentation_label)
                validation_segmentation_loss += segmentation_loss_value.item()
                loss = segmentation_loss_value

            
            if data["dt_label"][0]=='A2D2_box':
                inputs = data["image"].to(device=device) 
                segmentation_label = data["box"].to(device=device)
    
                steering_output ,segmentation_output,box_output = model(inputs)
    
                
    
                box_loss_value = box_loss(box_output, box_label)
                validation_box_loss += box_loss_value.item()
                loss = box_loss_value
    

            total_validation_loss += loss.item()
            
            

    avgValLoss = total_validation_loss / len(val_dataloader)
    avgValSteeringLoss = validation_steering_loss / len(A2D2_dataset_val_str)
    avgValSegmentationLoss = validation_segmentation_loss / len(A2D2_dataset_val_seg)
    avgValBoxLoss = validation_box_loss / len(A2D2_dataset_val_box)


    wandb.log({
               "Average Train Loss": avgTrainLoss,
               "Average Train Steering Loss": avgTrainSteeringLoss,
               "Average Train Segmentation Loss": avgTrainSegmentationLoss,
               "Average Train Box Loss": avgTrainBoxLoss,
               "Average Validation Loss": avgValLoss,
               "Average Validation Steering Loss": avgValSteeringLoss,
               "Average Validation Segmentation Loss": avgValSegmentationLoss,
               "Average Validation Box Loss": avgValBoxLoss
    })

    print('Epoch [{}/{}]\n'
          'Train Loss: {:.5f} | Train Steering Loss: {:.5f} | Train Segmentation Loss: {:.5f} | Train Box Loss: {:.5f} \n'
          'Validation Loss: {:.5f} | Validation Steering Loss: {:.5f} | Validation Segmentation Loss: {:.5f} Validation Box Loss: {:.5f}'
          .format(epoch + 1, n_epochs, avgTrainLoss, avgTrainSteeringLoss, avgTrainSegmentationLoss,avgTrainBoxLoss, 
                                       avgValLoss,   avgValSteeringLoss,   avgValSegmentationLoss,  avgValBoxLoss))
    
    if avgValLoss<best_val_loss:
        best_val_loss=avgValLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)



