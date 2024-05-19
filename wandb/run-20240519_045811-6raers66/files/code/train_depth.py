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
from datasets import A2D2_steering,A2D2_seg,A2D2_depth,a2d2_dataloader
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



A2D2_path_train,A2D2_path_val=a2d2_dataloader()


A2D2_dataset_train_seg=A2D2_depth(A2D2_path_train)
A2D2_dataset_val_seg=A2D2_depth(A2D2_path_val)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = A2D2_dataset_train_seg        
val_dataset = A2D2_dataset_val_seg  

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

                  "Training Depth Loss" : depth_loss_value
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


