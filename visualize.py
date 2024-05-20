import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np

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
import yaml
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
from PIL import Image
import matplotlib.pyplot as plt

from visualisers import visualize_seg,visualize_depth,visualize_box


torch.manual_seed(42)
random.seed(42)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['segmentation','boundingbox', 'depth'],required=True)
parser.add_argument('--sample',type=int,required=True)
parser.add_argument('--threshold',type=float)



args = parser.parse_args()

if args.task == 'boundingbox' and args.threshold is None:
    print("Error: --threshold is required when --task is 'boundingbox'")
    exit()



A2D2_path_train,A2D2_path_val=a2d2_dataloader()





if args.task == 'segmentation':
    train_dataset = A2D2_seg(A2D2_path_train)
    val_dataset = A2D2_seg(A2D2_path_val)
elif args.task == 'boundingbox':
    train_dataset = A2D2_box(A2D2_path_train)
    val_dataset = A2D2_box(A2D2_path_val)
elif args.task == 'depth':
    train_dataset = A2D2_depth(A2D2_path_train)
    val_dataset = A2D2_depth(A2D2_path_val)


sample_no=args.sample

if sample_no>=len(A2D2_path_val):
    print('')
    print('Sample does not exist, please choose integer smaller that ',len(A2D2_path_val))
    exit()


sample=val_dataset[sample_no]



directory = f"results/{args.task}/{sample_no}"

if not os.path.exists(directory):
    os.makedirs(directory)


reverse_transform = transforms.Compose([
    transforms.ToPILImage(),

])

real_image_file =   f"results/{args.task}/{sample_no}/_real.png"


ground_truth_file =   f"results/{args.task}/{sample_no}/_truth.png"

symmetric_result_name = f"results/{args.task}/{sample_no}/_symmetric.png"

asymmetric_result_name = f"results/{args.task}/{sample_no}/_asymmetric.png"

universal_result_name = f"results/{args.task}/{sample_no}/_universal.png"

single_task_result_name = f"results/{args.task}/{sample_no}/_single_task.png"


print("Truth: ")
if args.task == 'segmentation':
    real_image = sample['image']
    truth=sample['A2D2_seg'] 


    real_image=reverse_transform(real_image)
    real_image.save(real_image_file)

    visualize_seg(seg_output=truth,file_name=ground_truth_file)

elif args.task == 'depth':
    real_image = sample['image']
    truth=sample['A2D2_depth'] 


    real_image=reverse_transform(real_image)
    real_image.save(real_image_file)

    visualize_depth(truth, file_name=ground_truth_file)

elif args.task == 'boundingbox':
    image=sample['image']
    boxes_ground_truth=sample['A2D2_box']
    
    
    visualize_box(image, boxes_ground_truth, ground_truth_file, 1.0)


with open('./configs/default_weights_conf.yaml', 'r') as config_file:
    weight_config = yaml.safe_load(config_file)

print('Done!')
print("Asymmetric: ")
model=ResNETBiFPN()   
model.load_state_dict(torch.load(weight_config['asymmetric_weight']))


sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


if args.task == 'segmentation':
    real_image = sample['image']
    output = output[1][0]
    


    visualize_seg(seg_output=output,file_name=asymmetric_result_name)

elif args.task == 'depth':
    real_image = sample['image']
    truth=output[3][0]


    visualize_depth(truth, file_name=asymmetric_result_name)

elif args.task == 'boundingbox':
    image=sample['image']
    boxes_ground_truth=output[2][0]
    

    visualize_box(image, boxes_ground_truth, asymmetric_result_name, args.threshold)

print('Done!')
print("Symmetric: ")
model=ResNETBiFPN()   
model.load_state_dict(torch.load(weight_config['symmetric_weight']))


sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


if args.task == 'segmentation':
    real_image = sample['image']
    output = output[1][0]
    
    visualize_seg(seg_output=output,file_name=symmetric_result_name)

elif args.task == 'depth':
    real_image = sample['image']
    truth=output[3][0]


    visualize_depth(truth, file_name=symmetric_result_name)

elif args.task == 'boundingbox':
    image=sample['image']
    boxes_ground_truth=output[2][0]
    

    visualize_box(image, boxes_ground_truth, symmetric_result_name, args.threshold)

print('Done!')

print("Universal: ")
model=ResNETBiFPN()   
model.load_state_dict(torch.load(weight_config['universal_task_weight'][args.task]))


sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)


if args.task == 'segmentation':
    real_image = sample['image']
    output = output[1][0]
    

    visualize_seg(seg_output=output,file_name=universal_result_name)

elif args.task == 'depth':
    real_image = sample['image']
    output=output[3][0]


    visualize_depth(output, file_name=universal_result_name)

elif args.task == 'boundingbox':
    image=sample['image']
    boxes_ground_truth=output[2][0]
    

    visualize_box(image, boxes_ground_truth, universal_result_name, args.threshold)

print('Done!')
print("Single Task: ")


if args.task == 'segmentation':
    real_image = sample['image']
    model=UNet()
    model.load_state_dict(torch.load(weight_config['single_task_weight'][args.task]))
    with torch.no_grad():
        output = model(sample_input)



    visualize_seg(seg_output=output[0],file_name=single_task_result_name)

elif args.task == 'depth':
    real_image = sample['image']
    model=PTModel()
    model.load_state_dict(torch.load(weight_config['single_task_weight'][args.task]))
    with torch.no_grad():
        output = model(sample_input)


    visualize_depth(output[0], file_name=single_task_result_name)

elif args.task == 'boundingbox':
    real_image = sample['image']
    model=YOLOv1()
    model.load_state_dict(torch.load(weight_config['single_task_weight'][args.task]))
    with torch.no_grad():
        output = model(sample_input)
    

    visualize_box(image, output[0], single_task_result_name, args.threshold)
print('Done!')
