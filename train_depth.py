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
from datasets import A2D2_steering,A2D2_seg,A2D2_depth
from losses import YOLOLoss,DepthLoss,MaskedMSELoss,MaskedL1Loss

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.UNet_depth import UNet
from models.DenseDepth import PTModel
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
filename = f"weights/depth_{current_time}.pth"


BATCH_SIZE = 4

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

insult_list=['/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20180810_142822/camera/cam_front_center/20180810142822_camera_frontcenter_000000004.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000020.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000022.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000041.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000049.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000050.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000055.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000057.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000063.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000035.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000051.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000055.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000061.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181108_103155/camera/cam_front_center/20181108103155_camera_frontcenter_000000028.png',
             '/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/20181108_103155/camera/cam_front_center/20181108103155_camera_frontcenter_000000033.png']


insult_list = np.array(insult_list)
BASE_PATH = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/"

# Get a list of all folders in the base path
all_folders = sorted(os.listdir(BASE_PATH))
all_folders = all_folders[1:-3]
print(all_folders)
# exit()

A2D2_path_train_seg=[]
A2D2_path_train_str=[]

A2D2_path_val_seg = []
A2D2_path_val_str = [] 

A2D2_path_train  = [] 
A2D2_path_val = []

for folder in all_folders:
    folder_path = os.path.join(BASE_PATH, folder)
    
    # Get a list of all files in the current folder
    files_in_folder = np.array(sorted(glob.glob(os.path.join(folder_path, "camera/cam_front_center/*.png"))))
    files_in_folder = files_in_folder[~np.isin(files_in_folder, insult_list)]

    
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




A2D2_path_train_seg=A2D2_path_train[:int(len(A2D2_path_train)/2)]
A2D2_path_train_str=A2D2_path_train[int(len(A2D2_path_train)/2):]

A2D2_path_val_seg= A2D2_path_val[:int(len(A2D2_path_val)/2)]
A2D2_path_val_str= A2D2_path_val[int(len(A2D2_path_val)/2):]

A2D2_dataset_train_seg=A2D2_depth(A2D2_path_train_seg)
# A2D2_dataset_train_str=A2D2_steering(A2D2_path_train_str)

A2D2_dataset_val_seg=A2D2_depth(A2D2_path_val_seg)
# A2D2_dataset_val_str=A2D2_steering(A2D2_path_val_str)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = A2D2_dataset_train_seg    #ConcatDataset([A2D2_dataset_train_seg,A2D2_dataset_train_str])      #AllDatasets2(A2D2_dataset_train,Kitty_dataset_train,BATCH_SIZE)
val_dataset = A2D2_dataset_val_seg #  ConcatDataset([A2D2_dataset_val_seg,A2D2_dataset_val_str])      #AllDatasets2(A2D2_dataset_val,Kitty_dataset_val,BATCH_SIZE)


print('No of train samples', len(train_dataset))
print('No of validation Samples', len(val_dataset))




train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=24)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)

model = PTModel()
model.to(device=device)
depth_loss = DepthLoss()  #1 




lr = 1e-5
# # momentum = 0.9  
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_epochs=40

best_val_loss=999999
config = {
    "learning_rate": lr,
    # "depth_loss": depth_loss.__class__.__name__,
    # "steering_loss": steering_loss.__class__.__name__,
    # "batch_backbone": BATCH_SIZE
    # "batch_steering": BATCH_SIZE
    "batch_depth": BATCH_SIZE

    # "optimizer": optimizer.__class__.__name__,
    # "A2D2_dataset_train_size": len(A2D2_dataset_train),
    # "Kitty_dataset_train_size": len(Kitty_dataset_train),
    # "A2D2_dataset_val_size": len(A2D2_dataset_val),
    # "Kitty_dataset_val_size": len(Kitty_dataset_val)
}


run_name = f"depth_{current_time}"
wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)

for epoch in range(n_epochs):
    
    model.train()

    total_training_loss = 0
    training_steering_loss = 0
    training_depth_loss = 0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
        # model.train()        
        
        inputs = data["image"].to(device=device) 
        depth_label = data["A2D2_depth"].to(device=device)


        

        #Output
        optimizer.zero_grad()
        depth_output = model(inputs)

        # print(depth_output.shape)
        # print(depth_label.shape)
        # exit()


        # Loss calculation
        depth_loss_value = depth_loss(depth_output, depth_label)

        #Backward
        depth_loss_value.backward()
        optimizer.step()
        
        #Logging
        
        training_depth_loss += depth_loss_value


        wandb.log({
                #   "Training Loss ": loss,
                #   "Training Steering Loss ": steering_loss_value,
                  "Training Depth Loss" : depth_loss_value
                #   "Average Total Training Loss (During Epoch)": total_training_loss / (i + 1), 
                #   "Average Training Steering Loss (During Epoch)": training_steering_loss / (i + 1),
                #   "Average Training depth Loss (During Epoch)": training_depth_loss / (i + 1)
                })
        

    avgTrainLoss = total_training_loss / len(A2D2_dataset_train_seg)
    avgTrainDepthLoss = training_depth_loss / len(train_dataloader)

# #---------------------------------------------------------------------------------------------------------


    model.eval()
    total_validation_loss = 0

    validation_depth_loss = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            inputs = data["image"].to(device=device) 
            depth_label = data["A2D2_depth"].to(device=device)

            depth_output = model(inputs)
            depth_loss_value = depth_loss(depth_output, depth_label)


            
            validation_depth_loss += depth_loss_value.item()

    avgValLoss = total_validation_loss / len(val_dataloader)
    avgValDepthLoss = validation_depth_loss / len(val_dataloader)

    wandb.log({
            #    "Train Loss": avgTrainLoss,
            #    "Train Steering Loss": avgTrainSteeringLoss,
               "Average Train Depth Loss": avgTrainDepthLoss,
            #    "Validation Loss": avgValLoss,
            #    "Validation Steering Loss": avgValSteeringLoss,
               "Average Validation Depth Loss": avgValDepthLoss
    })




    # print('Epoch [{}/{}]\n'
    #       'Train Loss: {:.4f} | Train depth Loss: {:.4f}\n'
    #       'Validation Loss: {:.4f}  | Validation depth Loss: {:.4f}'
    #       .format(epoch + 1, n_epochs, avgTrainLoss, avgTrainDepthLoss, avgValLoss,
    #                avgValDepthLoss))
    if avgValDepthLoss<best_val_loss:
        best_val_loss=avgValDepthLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)

    
# torch.save(best_val.state_dict(), "multi_channel.pth")


