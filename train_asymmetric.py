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
import yaml
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
# from models.resnet_bifpn import ResNETBiFPN
from models.resnet_bifpn_depth import ResNETBiFPN
from losses import YOLOLoss,DepthLoss,MaskedMSELoss,MaskedL1Loss
import wandb
from datetime import datetime
import os

from trainer import AsymmetricTrainer

torch.manual_seed(42)
random.seed(42)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

base_name = "universal_asymmetric"




BATCH_SIZE = 1

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

cient for symmetric task


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



A2D2_dataset_train_seg=A2D2_seg(A2D2_path_train)
A2D2_dataset_train_str=A2D2_steering(A2D2_path_train)
A2D2_dataset_train_box=A2D2_box(A2D2_path_train)
A2D2_dataset_train_dep=A2D2_depth(A2D2_path_train)

A2D2_dataset_val_seg=A2D2_seg(A2D2_path_val)
A2D2_dataset_val_str=A2D2_steering(A2D2_path_val)
A2D2_dataset_val_box=A2D2_box(A2D2_path_val)
A2D2_dataset_val_dep=A2D2_depth(A2D2_path_val)



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




train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=35)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=35)


model = ResNETBiFPN()
model.to(device=device)

##############
##LOSSnPARAMS#
##############
segmentation_loss = nn.CrossEntropyLoss()
steering_loss = nn.L1Loss()  
box_loss=YOLOLoss()
depth_loss=DepthLoss()


backbone_params    =list(model.backbone.parameters())+list(model.backbone_1.parameters())+list(model.backbone_2.parameters())+list(model.backbone_3.parameters())+list(model.backbone_4.parameters())+list(model.channel_1.parameters())+list(model.channel_2.parameters())+list(model.channel_3.parameters())+list(model.channel_4.parameters())+list(model.bifpn.parameters())+list(model.bifpn_1.parameters())
segmentation_params=list(model.segmentation_head.parameters())
steering_params=list(model.steering_angle_backbone.parameters())+list(model.steering_angle_head_21.parameters())
box_params=list(model.box_head.parameters())
depth_params=list(model.depth_head.parameters())

##############
##LOSSnPARAMS#
##############

##############
##Parameters##
##############
# export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
with open('./asymmetric_conf.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

batch_sizes = config['batch_sizes']
learning_rates = config['learning_rates']
coefficients = config['coefficients']
n_epochs = config['n_epochs']


batch_backbone = batch_sizes['batch_backbone']
batch_steering = batch_sizes['batch_steering']
batch_segmentation = batch_sizes['batch_segmentation']
batch_box = batch_sizes['batch_box']
batch_depth = batch_sizes['batch_depth']
backbone_lr = learning_rates['backbone_lr']
segmentation_lr = learning_rates['segmentation_lr']
steering_lr = learning_rates['steering_lr']
box_lr = learning_rates['box_lr']
depth_lr = learning_rates['depth_lr']
steering_coef = coefficients['steering_coef']
segmentation_coef = coefficients['segmentation_coef']
box_coef = coefficients['box_coef']
depth_coef = coefficients['depth_coef']

print("Batch sizes:")
print(f"  Backbone: {batch_backbone}")
print(f"  Steering: {batch_steering}")
print(f"  Segmentation: {batch_segmentation}")
print(f"  Box: {batch_box}")
print(f"  Depth: {batch_depth}")

print("Learning rates:")
print(f"  Backbone LR: {backbone_lr}")
print(f"  Segmentation LR: {segmentation_lr}")
print(f"  Steering LR: {steering_lr}")
print(f"  Box LR: {box_lr}")
print(f"  Depth LR: {depth_lr}")

print("Coefficients:")
print(f"  Steering Coef: {steering_coef}")
print(f"  Segmentation Coef: {segmentation_coef}")
print(f"  Box Coef: {box_coef}")
print(f"  Depth Coef: {depth_coef}")

print(f"Number of epochs: {n_epochs}")
##############
##Parameters##
##############


backbone_optimizer = torch.optim.Adam(backbone_params, lr=backbone_lr/batch_backbone)
segmentation_optimizer = torch.optim.Adam(segmentation_params, lr=segmentation_lr)
steering_optimizer = torch.optim.Adam(steering_params, lr=steering_lr)
box_optimizer = torch.optim.Adam(box_params, lr=box_lr)
depth_optimizer= torch.optim.Adam(depth_params, lr=depth_lr)



best_val_loss=999999

config = {
    "learning_rate": backbone_lr,
    "segmentation_learning_rate": segmentation_lr,
    "steering_learning_rate": steering_lr,
    "box_learning_rate": box_lr,
    "depth_learning_rate": depth_lr,

    "batch_backbone": batch_backbone,
    "batch_steering": batch_steering,
    "batch_box": batch_box,
    "batch_segmentation":batch_segmentation,
    "batch_depth": batch_depth
}


steering_trainer = AsymmetricTrainer(optimizer=steering_optimizer, batch_size=batch_steering ,loss_func=steering_loss, device=device, wandb_log='Steering',
                                     grad_dec_coef=steering_coef,data_len_tr=len(A2D2_dataset_train_str),data_len_val=len(A2D2_dataset_val_str))

box_trainer = AsymmetricTrainer(optimizer=box_optimizer, batch_size=batch_box ,loss_func=box_loss, device=device, wandb_log='Boxes',
                                grad_dec_coef=box_coef,data_len_tr=len(A2D2_dataset_train_box),data_len_val=len(A2D2_dataset_val_box))

segmentation_trainer = AsymmetricTrainer(optimizer=segmentation_optimizer, batch_size=batch_segmentation ,loss_func=segmentation_loss, device=device, wandb_log='Segmentation',
                                         grad_dec_coef=segmentation_coef,data_len_tr=len(A2D2_dataset_train_seg),data_len_val=len(A2D2_dataset_val_seg))


depth_trainer = AsymmetricTrainer(optimizer=depth_optimizer, batch_size=batch_depth ,loss_func=depth_loss, device=device, wandb_log='Depth',
                                         grad_dec_coef=depth_coef,data_len_tr=len(A2D2_dataset_train_dep),data_len_val=len(A2D2_dataset_val_dep))




filename = f"weights/{base_name}_{current_time}.pth"
run_name = f"{base_name}_{current_time}"

wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)
for epoch in range(n_epochs):

    model.train()

    backbone_counter=0


    total_tr_loss=0
    batch_tr_loss=0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):

        if data["dt_label"][0] == 'A2D2_steering':
            backbone_counter=backbone_counter+1   
            model,steering_loss_item =steering_trainer.train_batch(model, data)

            total_tr_loss+=steering_loss_item
            batch_tr_loss+=steering_loss_item
    
        if data["dt_label"][0] == 'A2D2_box':
            backbone_counter=backbone_counter+1   
            model,box_loss_item =box_trainer.train_batch(model, data)

            total_tr_loss+=box_loss_item
            batch_tr_loss+=box_loss_item

    
        if data["dt_label"][0] == 'A2D2_seg':
            backbone_counter=backbone_counter+1   
            model,segmentation_loss_item =segmentation_trainer.train_batch(model, data)
             
            total_tr_loss+=segmentation_loss_item
            batch_tr_loss+=segmentation_loss_item

        if data["dt_label"][0] == 'A2D2_depth':
            backbone_counter=backbone_counter+1   
            model,depth_loss_item =depth_trainer.train_batch(model, data)
             
            total_tr_loss+=depth_loss_item
            batch_tr_loss+=depth_loss_item


    
        


        if backbone_counter%batch_backbone==0:
            backbone_optimizer.step()
            backbone_optimizer.zero_grad()
            backbone_counter=0
            wandb.log({"Training Loss": batch_tr_loss/batch_backbone})
            batch_tr_loss=0




# #---------------------------------------------------------------------------------------------------------


    model.eval()

    total_val_loss=0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):        

            if data["dt_label"][0]=='A2D2_steering':
                steering_loss_item =steering_trainer.validate_batch(model, data)
                total_val_loss=total_val_loss+steering_loss_item
            

                
            if data["dt_label"][0]=='A2D2_seg':
                segmentation_loss_item =segmentation_trainer.validate_batch(model, data)
                total_val_loss=total_val_loss+segmentation_loss_item
            
            if data["dt_label"][0]=='A2D2_box':
                box_loss_item =box_trainer.validate_batch(model, data)
                total_val_loss=total_val_loss+box_loss_item

            if data["dt_label"][0]=='A2D2_depth':
                depth_loss_item =depth_trainer.validate_batch(model, data)
                total_val_loss=total_val_loss+depth_loss_item
            
            

    avgValLoss = total_val_loss / len(A2D2_dataset_val_seg)
    avgTrainLoss = total_tr_loss / len(A2D2_dataset_train_seg)


    wandb.log({
               "Average Train Loss": avgTrainLoss,
               "Average Validation Loss": avgValLoss

    })


    if avgValLoss<best_val_loss:
        best_val_loss=avgValLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)

    segmentation_trainer.end_epoch()
    box_trainer.end_epoch()
    steering_trainer.end_epoch()
    depth_trainer.end_epoch()

    torch.cuda.empty_cache()




