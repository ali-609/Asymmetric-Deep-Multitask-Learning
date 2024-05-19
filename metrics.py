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
from datasets import A2D2_steering,A2D2_seg,A2D2_box,A2D2_depth,a2d2_dataloader

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
from losses import YOLOLoss,MaskedL1Loss
import wandb
from datetime import datetime
import os
import argparse


torch.manual_seed(42)
random.seed(42)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--data', choices=['segmentation', 'steering', 'box', 'depth'])
parser.add_argument('--model', choices=['PilotNet', 'UNet', 'YOLO', 'DenseDepth','MTL'])
parser.add_argument('--weights')


args = parser.parse_args()

BATCH_SIZE = 1

A2D2_path_train,A2D2_path_val=a2d2_dataloader()







if args.data == 'segmentation':
    train_dataset = A2D2_seg(A2D2_path_train)
    val_dataset = A2D2_seg(A2D2_path_val)
elif args.data == 'steering':
    train_dataset = A2D2_steering(A2D2_path_train)
    val_dataset = A2D2_steering(A2D2_path_val)
elif args.data == 'box':
    train_dataset = A2D2_box(A2D2_path_train)
    val_dataset = A2D2_box(A2D2_path_val)
elif args.data == 'depth':
    train_dataset = A2D2_depth(A2D2_path_train)
    val_dataset = A2D2_depth(A2D2_path_val)




print('No of total train samples', len(train_dataset))
print('No of total validation Samples', len(val_dataset),'\n')



# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=10)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)


if args.model == 'MTL':
    model = ResNETBiFPN()
    model.load_state_dict(torch.load('weights/universal_symmetric_2024-05-02_18-34-33.pth'))
elif args.model == 'UNet':
    model = UNet()
    model.load_state_dict(torch.load('weights/segmentation_2024-04-21_00-16-07.pth'))
elif args.model == 'YOLO':
    model = YOLOv1()
    model.load_state_dict(torch.load('weights/yolo_2024-05-03_19-36-16.pth'))
elif args.model == 'DenseDepth':
    model = PTModel()
    model.load_state_dict(torch.load('weights/depth_2024-04-24_01-22-43.pth'))
elif args.model == 'PilotNet':
    model = PilotNet()
    model.load_state_dict(torch.load('weights/steering_2024-04-25_04-27-04.pth'))

if args.weights:
    model.load_state_dict(torch.load(args.weights))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device)



class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss_func=nn.L1Loss()

    def forward(self, pred, target):
        non_zero_mask = target != 0

        # Masked predictions and targets
        masked_predictions = pred[non_zero_mask]
        masked_targets = target[non_zero_mask]
        

        self.loss = self.loss_func(masked_predictions,masked_targets)
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



 
                                 
        return average_max_iou



class SegmentationAccuracy(nn.Module):
    def __init__(self):
        super(SegmentationAccuracy, self).__init__()

    def forward(self,pred,target):
        pred_values_max,pred_indices_max=torch.max(pred,dim=1)

        target_values_max,target_indices_max=torch.max(target,dim=1)


        TP=(pred_indices_max==target_indices_max).sum()

        acc=TP/(1024*1024)



        
        return acc

if args.data == 'segmentation':
    task_metric = SegmentationAccuracy()
elif args.data == 'steering':
    task_metric = SteeringMetric()
elif args.data == 'box':
    task_metric = BoxAccuracy()
elif args.data == 'depth':
    task_metric = MaskedL1Loss()



total_met=0.0

with torch.no_grad():
    for i, data in enumerate(tqdm(val_dataloader)):
        inputs = data["image"].to(device=device)
        # print(inputs.shape)
        # exit()


        
        if args.data == 'segmentation':
            labels = data["A2D2_seg"].to(device=device)
            if args.model == 'MTL':
                steering_output, segmentation_output, box_output, depth_output = model(inputs)
            else:
                 segmentation_output=model(inputs)

            metric_value = task_metric(segmentation_output, labels)
        elif args.data == 'steering':
            labels = data["A2D2_steering"].to(device=device)

            if args.model == 'MTL':
                steering_output, segmentation_output, box_output, depth_output = model(inputs)
            else:
                 steering_output=model(inputs)

            metric_value = task_metric(labels, steering_output)

        elif args.data == 'box':
            labels = data["A2D2_box"].to(device=device)

            if args.model == 'MTL':
                steering_output, segmentation_output, box_output, depth_output = model(inputs)
            else:
                 box_output=model(inputs)

            metric_value = task_metric(box_output, labels)
        elif args.data == 'depth':
            labels = data["A2D2_depth"].to(device=device)

            if args.model == 'MTL':
                steering_output, segmentation_output, box_output, depth_output = model(inputs)
            else:
                depth_output=model(inputs)

            metric_value = task_metric(depth_output, labels)
        
        total_met += metric_value



          

final_metric=total_met/len(val_dataloader)

print(final_metric)

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