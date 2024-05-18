import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

import json
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os
from skimage.util import random_noise
import glob
from scipy.spatial.distance import cdist
from PIL import Image
import re
from multiprocessing import Pool

files=sorted(glob.glob(        "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/2018*/label2D/cam_front_center/*.json"))
A2D2_path_all=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))

print(len(files))
print(len(A2D2_path_all))

width_scale=256/1920
height_scale=256/1208

# exit()
max_val=0
min_val=212
sum=0
max_boxes_in_grid = 0

object_classes=['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']
unique_classes = []
for file in files:
    out_path = file.replace('.json','.npy')
    output=np.zeros((8, 8, 399))

    filel=open(file)
    
    data=json.load(filel)
    boxes_in_grid_counter = np.zeros((8, 8),dtype=int)

    for i, items in enumerate(data):

        class_name=data[items]['class']
        class_index = object_classes.index(class_name)

        #box position and size
        coordinates=data[items]['2d_bbox']
        width=(coordinates[0]+coordinates[2])/2
        height=(coordinates[1]+coordinates[3])/2

        width_size=coordinates[2]-coordinates[0]
        height_size=coordinates[3]-coordinates[1]

        #gridding and scaling
        grid_width=int(width//240)
        grid_height=int(height//151)

        ingrid_width=(width%240)*width_scale
        ingrid_height=(height%151)*height_scale

        scaled_width_size = width_size*width_scale
        scaled_height_size = height_size*height_scale

        grid_height = np.clip(grid_height, 0, 7)
        grid_width = np.clip(grid_width, 0, 7)

        #putting everything together
        output[grid_height, grid_width,i*19]=1 # confidence
        output[grid_height, grid_width, i*19 + 1] = ingrid_width  # x-coordinate within the grid
        output[grid_height, grid_width, i*19 + 2] = ingrid_height  # y-coordinate within the grid
        output[grid_height, grid_width, i*19 + 3] = scaled_width_size  # width
        output[grid_height, grid_width, i*19 + 4] = scaled_height_size
        output[grid_height, grid_width, i*19 + 5 + class_index] = 1 # class encoding

        boxes_in_grid_counter[grid_height, grid_width] += 1

        # max_boxes_in_grid = max(max_boxes_in_grid, int(np.sum(output[:,:,i*19])))
        if boxes_in_grid_counter[grid_height, grid_width]>max_boxes_in_grid:
            max_boxes_in_grid=boxes_in_grid_counter[grid_height, grid_width]
            print('Howdy, There are ',max_boxes_in_grid,' boxes in grid ',grid_height, grid_width)
            print('File name: ',file)
            print(boxes_in_grid_counter)
        # max_boxes_in_grid = max(max_boxes_in_grid, boxes_in_grid_counter[grid_height, grid_width])







        # print(class_name)
        # if class_name not in unique_classes:
            # unique_classes.append(class_name)
            # print('New class found: ',class_name)
        
        # exit()
        # class_name=items.get('class')
    # print(class_name)
    
    # exit()



    sum+=len(data.keys())
    if len(data.keys())>max_val:
        max_val=len(data.keys())
    if len(data.keys())<min_val:
        min_val=len(data.keys())
print("Maximum number of boxes in same grid:", max_boxes_in_grid)
# print('Avaible classes are: ', unique_classes)
# print('Number of classes are: ', len(unique_classes))
print('Average number of boxes: ',sum/len(files))
print('Maximum number of boxes per frame: ',max_val)
print('Minimum number of boxes per frame: ',min_val)