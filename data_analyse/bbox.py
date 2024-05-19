import sys
from pathlib import Path


import numpy as np

import json

import os

import glob

from PIL import Image
import re
from multiprocessing import Pool

import argparse

# Define and parse command-line arguments
parser = argparse.ArgumentParser()


rel_dirs=sorted(glob.glob("./Datasets/camera_lidar_semantic_bboxes/2018*/label2D/cam_front_center/*.json"))


files = [os.path.abspath(path) for path in rel_dirs]

print(len(files))


width_scale=256/1920
height_scale=256/1208

# exit()
max_val=0
min_val=212
sum=0
max_boxes_in_grid = 0


parser.add_argument('--grid-setup', nargs=2, type=int, default=[8, 8], help='Grid setup dimensions')

args = parser.parse_args()

grid_setup = args.grid_setup

anchor_box=21

grid_size_x=(1920/grid_setup[0])
grid_size_y=(1208/grid_setup[1])

object_classes=['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']
unique_classes = []
for file in files:
    out_path = file.replace('.json','.npy')
    output=np.zeros((grid_setup[0], grid_setup[1], 19*anchor_box))

    filel=open(file)
    
    data=json.load(filel)
    boxes_in_grid_counter = np.zeros((grid_setup[0], grid_setup[1]),dtype=int)

    for i, items in enumerate(data):

        class_name=data[items]['class']
        class_index = object_classes.index(class_name)

        #box position and size
        coordinates=data[items]['2d_bbox']
        box_center_x=(coordinates[0]+coordinates[2])/2
        box_center_y=(coordinates[1]+coordinates[3])/2

        box_width=coordinates[2]-coordinates[0]
        box_height=coordinates[3]-coordinates[1]
    
        grid_x=int(box_center_x//grid_size_x) #240)
        grid_y=int(box_center_y//grid_size_y) #151)

        if grid_x>7:
            box_width=box_width-(1920/grid_setup[0])*(abs(grid_x-7)*2)
                    # print('after',box_width,'-',abs(grid_x-7),'==',grid_x)
    
    
        if grid_x<0:
            box_width=box_width-(1920/grid_setup[1])*(abs(grid_x-0)*2)
                    # print('after',box_width,'-',abs(grid_x-0),'==',grid_x)
    
        ingrid_x=(box_center_x%grid_size_x)/grid_size_x
        ingrid_y=(box_center_y%grid_size_y)/grid_size_y
    
    
        scaled_box_width =  box_width/grid_size_x
        scaled_box_height = box_height/grid_size_y
    
    
    
        grid_x = np.clip(grid_x, 0, grid_setup[0]-1)
        grid_y = np.clip(grid_y, 0, grid_setup[1]-1)
                
                
        #putting everything together
        output[grid_x, grid_y,i*19]=1 # confidence
        output[grid_x, grid_y, i*19 + 1] = ingrid_x #coordinate within the grid
        output[grid_x, grid_y, i*19 + 2] = ingrid_y #coordinate within the grid
        output[grid_x, grid_y, i*19 + 3] = scaled_box_width # # width
        output[grid_x, grid_y, i*19 + 4] = scaled_box_height #
        output[grid_x, grid_y, i*19 + 5 + class_index] = 1 # class encoding

        boxes_in_grid_counter[grid_x, grid_y] += 1

        # max_boxes_in_grid = max(max_boxes_in_grid, int(np.sum(output[:,:,i*19])))
        if boxes_in_grid_counter[grid_x, grid_y]>max_boxes_in_grid:
            max_boxes_in_grid=boxes_in_grid_counter[grid_x, grid_y]
            print('There are ',max_boxes_in_grid,' boxes in grid ',grid_x, grid_y)
            print('File name: ',file)
            print(boxes_in_grid_counter)




        if class_name not in unique_classes:
            unique_classes.append(class_name)
            print('New class found: ',class_name)
        




    sum+=len(data.keys())
    if len(data.keys())>max_val:
        max_val=len(data.keys())
    if len(data.keys())<min_val:
        min_val=len(data.keys())
print("Maximum number of boxes in same grid:", max_boxes_in_grid)
print('Avaible classes are: ', unique_classes)
print('Number of classes are: ', len(unique_classes))
print('Average number of boxes: ',sum/len(files))
print('Maximum number of boxes per frame: ',max_val)
print('Minimum number of boxes per frame: ',min_val)