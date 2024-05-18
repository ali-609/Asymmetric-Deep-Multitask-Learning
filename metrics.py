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
# import torchvision.ops.box_iou as IOU_F 

import glob
from datasets import A2D2_steering,A2D2_seg,A2D2_box,A2D2_depth

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
# from models.first_hydra import HydraNet
# from models.model_withUnet import HydraUNet
# from models.dino_backbone import DinoBackBone
# from models.resnet_bifpn import ResNETBiFPN
from models.resnet_bifpn_depth import ResNETBiFPN
from models.UNet import UNet
from models.yolo_v1 import YOLOv1
from models.DenseDepth import PTModel
from models.pilotnet import PilotNet
from losses import YOLOLoss,DepthLoss,MaskedMSELoss,MaskedL1Loss
import wandb
from datetime import datetime
import os

from trainer import AsymmetricTrainer

torch.manual_seed(42)
random.seed(42)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

base_name = "asymmetric_three_task"




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

train_dataset = A2D2_dataset_train_box #ConcatDataset([A2D2_dataset_train_seg,A2D2_dataset_train_str,A2D2_dataset_train_box,A2D2_dataset_train_dep])      
val_dataset =   A2D2_dataset_val_box#ConcatDataset([A2D2_dataset_val_seg,A2D2_dataset_val_str,A2D2_dataset_val_box,A2D2_dataset_val_dep])      


print('No of total train samples', len(train_dataset))
print('No of total validation Samples', len(val_dataset),'\n')

print('No of segmentation train samples', len(A2D2_dataset_train_seg))
print('No of segmentation validation Samples', len(A2D2_dataset_val_seg),'\n')


print('No of steering train samples', len(A2D2_dataset_train_str))
print('No of steering validation Samples', len(A2D2_dataset_val_str),'\n')

print('No of box train samples', len(A2D2_dataset_train_box))
print('No of box validation Samples', len(A2D2_dataset_val_box),'\n')




# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=10)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)


model = ResNETBiFPN()
model.load_state_dict(torch.load('weights/universal_symmetric_2024-05-02_18-34-33.pth'))

# model = UNet()
# model.load_state_dict(torch.load('weights/segmentation_2024-04-21_00-16-07.pth'))

# model = YOLOv1()
# model.load_state_dict(torch.load('weights/yolo_2024-05-03_19-36-16.pth' ))

# model=PTModel()
# model.load_state_dict(torch.load('weights/depth_2024-04-24_01-22-43.pth' ))

# model = PilotNet()
# model.load_state_dict(torch.load('weights/steering_2024-04-25_04-27-04.pth'))

model.to(device=device)



class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        self.loss = self.loss * 123
        return self.loss


class SteeringMetric(nn.Module):
        def __init__(self):
             super(SteeringMetric, self).__init__()
             self.loss_func=L1Loss()
        def forward(self,pred,target):
             loss_value=self.loss_func(pred,target)
             loss_value=loss_value*70

             return loss_value



class BoxAccuracy(nn.Module):
    def __init__(self):
        self.num_boxes=10

        super(BoxAccuracy, self).__init__()


    def forward(self,pred,target):
        target_confidence = target[0,0: self.num_boxes*19+0:19]
        target_exist=torch.nonzero(target_confidence)
        target_exist_num=len(target_exist)



        pred_confidence = pred[0,0: self.num_boxes*19+0:19]
        pred_confidence_flat=pred_confidence.flatten()
        sorted_confidences,_=torch.sort(pred_confidence_flat, descending=True)


        threshold= sorted_confidences[target_exist_num-1]


        pred_exist=torch.nonzero(torch.ge(pred_confidence, threshold))


        target_coords = torch.zeros((len(target_exist), 4))
        pred_coords = torch.zeros((len(target_exist), 4))


        for idx, box in enumerate(target_exist):
            grid_x=box[1]
            grid_y=box[2]
            nth_anchor=box[0]
            boxes=target[0]

            x =  grid_x * (1920/8)+ boxes[nth_anchor*19+1, grid_x, grid_y] * (1920/8)  # x-coordinate within the grid
            y =  grid_y * (1208/8)+ boxes[nth_anchor*19+2, grid_x, grid_y] * (1208/8)  # y-coordinate within the grid
            width = boxes[nth_anchor*19+3, grid_x, grid_y] * (1920 /8) # Width of the box
            height = boxes[nth_anchor*19+4, grid_x, grid_y] * (1208/8)  # Height of the box
                    # Convert center coordinates to top-left coordinates
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            target_coords[idx] = torch.tensor([x1, y1, x2, y2])


        for idx, box in enumerate(pred_exist):
            grid_x=box[1]
            grid_y=box[2]
            nth_anchor=box[0]
            boxes=pred[0]
            
            x =  grid_x * (1920/8)+ boxes[nth_anchor*19+1, grid_x, grid_y] * (1920/8)  # x-coordinate within the grid
            y =  grid_y * (1208/8)+ boxes[nth_anchor*19+2, grid_x, grid_y] * (1208/8)  # y-coordinate within the grid
            width = boxes[nth_anchor*19+3, grid_x, grid_y] * (1920 /8) # Width of the box
            height = boxes[nth_anchor*19+4, grid_x, grid_y] * (1208/8)  # Height of the box
                    # Convert center coordinates to top-left coordinates
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            if idx>=len(pred_coords):
                 break

            pred_coords[idx] = torch.tensor([x1, y1, x2, y2])

            

        iou=torchvision.ops.box_iou(target_coords,pred_coords)

        max_per_iou, _ = torch.max(iou, dim=1)


        average_max_iou = torch.mean(max_per_iou)



 
                                 
        return average_max_iou,target_exist_num
    


class SegmentationAccuracy(nn.Module):
    def __init__(self):
        super(SegmentationAccuracy, self).__init__()

    def forward(self,pred,target):
        pred_values_max,pred_indices_max=torch.max(pred,dim=1)

        target_values_max,target_indices_max=torch.max(target,dim=1)

        # pred_indices_max==target_values_max
        TP=(pred_indices_max==target_indices_max).sum()

        acc=TP/(1024*1024)

        # print(TP)

        
        return acc

seg_metric=SegmentationAccuracy()
box_metric=BoxAccuracy()
depth_metric=MaskedL1Loss()
str_metric=SteeringMetric()

total_box_met=0.0
total_seg_met=0.0
total_depth_met=0.0
total_str_met=0.0
# with torch.no_grad():
#         for i, data in enumerate(tqdm(val_dataloader)):
#             inputs = data["image"].to(device=device) 
#             segmentation_label = data["A2D2_seg"].to(device=device)

#             # segmentation_output = model(inputs)
#             steering_output, segmentation_output, box_output, depth_output = model(inputs)
#             # print(segmentation_label)
#             # print(segmentation_output)

#             seg_acc=seg_metric(segmentation_output,segmentation_label)

#             total_seg_met=total_seg_met+seg_acc


            # exit()

with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader)):
            inputs = data["image"].to(device=device) 
            box_label = data["A2D2_box"].to(device=device)

            # box_output = model(inputs)
            steering_output, segmentation_output, box_output, depth_output = model(inputs)
            # print(box_label.shape)
            # print(box_output.shape)

            # exit()

            box_acc_val,target_exist_num=box_metric(box_output,box_label)

            if box_acc_val>0.5 and target_exist_num>1:
                 print(i,'----',box_acc_val)

            total_box_met=total_box_met+box_acc_val
            if i%1000==0:
                 print(total_box_met/i)


# with torch.no_grad():
#         for i, data in enumerate(tqdm(val_dataloader)):
#             inputs = data["image"].to(device=device) 
#             depth_label = data["A2D2_depth"].to(device=device)

#             # depth_output = model(inputs)
#             steering_output, segmentation_output, box_output, depth_output = model(inputs)
#             # print(segmentation_label)
#             # print(segmentation_output)

#             depth_metric_val=depth_metric(depth_output,depth_label)

#             total_depth_met=total_depth_met+depth_metric_val

# with torch.no_grad():
#         for i, data in enumerate(tqdm(val_dataloader)):
#             inputs = data["image"].to(device=device) 
#             steering_label = data["A2D2_steering"].to(device=device)

#             # steering_output = model(inputs)
#             steering_output, segmentation_output, box_output, depth_output = model(inputs)
#             # print(segmentation_label)
#             # print(segmentation_output)

#             str_metric_val=str_metric(steering_label,steering_output)

#             total_str_met=total_str_met+str_metric_val

final_seg_acc=total_seg_met/len(val_dataloader)
final_depth_metric=total_depth_met/len(val_dataloader)
final_str_metric=total_str_met/len(val_dataloader)


print(final_seg_acc)

# final_box_metric=total_box_met/len(val_dataloader)

print(final_depth_metric)
# print(final_box_metric)
print(final_str_metric)

### Segmentation
# Universal single task: 92.99%
# Single task: 90.46%
# Asymmetric: 90.40%
# Symmetric: 82.43%


### Bunding Box
# Asymmetric: 27%
# Single task: 19.43%
# Symmetric: 19.09%
# Universal signle task: 2%

### Depth
# Universal single task: 6.7601 meter
# Symmetric: 7.0289 meter
# Asymmetric: 7.0401 meter
# Single task: 7.1215 meter

### Steering
# Universal single task: 0.0116 degree
# Single task: 0.0193 degree
# Symmetric: 0.0828
# Asymmetric: 0.5226 degree