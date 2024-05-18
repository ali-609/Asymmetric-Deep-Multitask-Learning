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
from datasets import A2D2_steering,A2D2_seg


import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.steering_onlyUNet import HydraUNetSTR
from models.pilotnet import PilotNet
# from UNet import UNet
import os
from datetime import datetime
import wandb
#RSync
#Sbatch
torch.manual_seed(42)
# wandb.init(project="ITS")
# torch.cuda.set_per_process_memory_fraction(0.9)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

# Construct the filename with the current time
filename = f"weights/steering_{current_time}.pth"


BATCH_SIZE = 8

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# A2D2_path_all=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))
# # Kitty_path_all=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/Kitty/semantics/training/image_2/*png"))

# A2D2_path_train=A2D2_path_all[:int(len(A2D2_path_all) * TRAIN_SPLIT)]
# A2D2_path_val=A2D2_path_all[-int(len(A2D2_path_all) * VAL_SPLIT):]


BASE_PATH_BOX = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/" 


all_folders_box = sorted(os.listdir(BASE_PATH_BOX))
# all_folders_box = all_folders_box[0:-2]
all_folders_box = all_folders_box[1:-3]


A2D2_path_train_seg=[]
A2D2_path_train_str=[]
A2D2_path_train=[]

A2D2_path_val_seg = []
A2D2_path_val_str = [] 
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



A2D2_dataset_train=A2D2_steering(A2D2_path_train)
A2D2_dataset_val=A2D2_steering(A2D2_path_val)


# A2D2_dataset_val=DatasetA2D2(A2D2_path_val)

# Kitty_dataset_train=DatasetKitty(Kitty_path_train)
# Kitty_dataset_val=DatasetKitty(Kitty_path_val)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = A2D2_dataset_train   #ConcatDataset([A2D2_dataset_train_seg,A2D2_dataset_train_str])      #AllDatasets2(A2D2_dataset_train,Kitty_dataset_train,BATCH_SIZE)
val_dataset = A2D2_dataset_val #  ConcatDataset([A2D2_dataset_val_seg,A2D2_dataset_val_str])      #AllDatasets2(A2D2_dataset_val,Kitty_dataset_val,BATCH_SIZE)


print('No of train samples', len(train_dataset))
print('No of validation Samples', len(val_dataset))




train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=24)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)


# model = HydraUNetSTR()
model = PilotNet()
# model.load_state_dict(torch.load('weights/model.pth' ))
model.to(device=device)
# # Loss functions
# segmentation_loss = nn.CrossEntropyLoss()#reduction='none')  #nn.MSELoss()#nn.BCELoss()#nn.CrossEntropyLoss()  #1 
steering_loss = nn.L1Loss()  #2

# backbone_params    =list(model.inc.parameters())+list(model.down1.parameters())+list(model.down2.parameters())+list(model.down3.parameters())+list(model.down4.parameters())
# segmentation_params=list(model.up1.parameters())+list(model.up2.parameters())+  list(model.up3.parameters())+  list(model.up4.parameters())+  list(model.outc.parameters())
# steering_params=list(model.steering_angle_backbone.parameters())+list(model.steering_angle_head_21.parameters())
# print(list(model.inc.parameters()))
# exit()

batch_backbone=24
batch_steering=8
batch_segmentation=16

lr = 1e-5
# # momentum = 0.9  
# backbone_optimizer = torch.optim.Adam(backbone_params, lr=lr/batch_backbone)
# segmentation_optimizer = torch.optim.Adam(segmentation_params, lr=lr/batch_segmentation)
# steering_optimizer = torch.optim.Adam(steering_params, lr=lr/batch_steering)
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
#Check relation between batch size,loss coef and learning rate

n_epochs=40

best_val_loss=999999

config = {
    "learning_rate": lr,
    # "segmentation_loss": segmentation_loss.__class__.__name__,
    # "steering_loss": steering_loss.__class__.__name__,
    # "batch_backbone": batch_backbone,
    "batch_steering": batch_steering,
    # "batch_segmentation":batch_segmentation

    # "optimizer": optimizer.__class__.__name__,
    # "A2D2_dataset_train_size": len(A2D2_dataset_train),
    # "Kitty_dataset_train_size": len(Kitty_dataset_train),
    # "A2D2_dataset_val_size": len(A2D2_dataset_val),
    # "Kitty_dataset_val_size": len(Kitty_dataset_val)
}


run_name = f"steering_{current_time}"
wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)
for epoch in range(n_epochs):

    model.train()

    training_steering_loss = 0


    backbone_counter=0
    segmentation_counter=0
    steering_counter=0


    steering_loss_value=0


    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
 
        

        inputs = data["image"].to(device=device) 
        steering_label = data["A2D2_steering"].to(device=device)


        steering_output = model(inputs)

        steering_loss_value = steering_loss(steering_output, steering_label)
        # exit()


            
        steering_loss_value.backward()



        optimizer.step()
        optimizer.zero_grad()


        
        

        training_steering_loss += steering_loss_value

        
        # print(training_segmentation_loss)
        wandb.log({

                  "Training Steering Loss " : steering_loss_value,


                #   "Average Training Steering Loss (During Epoch)": training_steering_loss / (i + 1),

                })
        




    avgTrainSteeringLoss = training_steering_loss / len(train_dataloader)


# #---------------------------------------------------------------------------------------------------------
# class custom()

    model.eval()
    total_validation_loss = 0
    validation_steering_loss = 0
    validation_segmentation_loss = 0

    backbone_counter=0
    segmentation_counter=0
    steering_counter=0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):    
            
            inputs = data["image"].to(device=device) 
            steering_label = data["A2D2_steering"].to(device=device)
    
    
            steering_output  = model(inputs)
    
            steering_loss_value = steering_loss(steering_output, steering_label)

    


            validation_steering_loss += steering_loss_value.item()



    avgValSteeringLoss = validation_steering_loss / len(val_dataloader)



    wandb.log({

               "Average Train Steering Loss" : avgTrainSteeringLoss,

               "Average Validation Steering Loss" : avgValSteeringLoss,

    })

    print('Epoch [{}/{}]\n'
          'Train Steering Loss: {:.4f} | Validation Steering Loss: {:.4f} '
          .format(epoch + 1, n_epochs, avgTrainSteeringLoss, avgValSteeringLoss))
    if avgValSteeringLoss<best_val_loss:
        best_val_loss=avgValSteeringLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)

    
# torch.save(best_val.state_dict(), "multi_channel.pth")


