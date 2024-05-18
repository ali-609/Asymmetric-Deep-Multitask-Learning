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

# A2D2_path_all_seg=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/label/cam_front_center/*.png"))
A2D2_path_all_seg=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/2018*/label/cam_front_center/*.png"))

colors=np.array([
    # Car Colors
[255, 0, 0],  # Car 1
[200, 0, 0],  # Car 2  #1
[150, 0, 0],  # Car 3  #2
[128, 0, 0],  # Car 4  #3

# Bicycle Colors
[182, 89, 6],  # Bicycle 1
[150, 50, 4],  # Bicycle 2   #5
[90, 30, 1],  # Bicycle 3   #6
[90, 30, 30],  # Bicycle 4   #7

# Pedestrian Colors
[204, 153, 255],  # Pedestrian 1
[189, 73, 155],  # Pedestrian 2   #9
[239, 89, 191],  # Pedestrian 3   #10

# Truck Colors
[255, 128, 0],  # Truck 1
[200, 128, 0],  # Truck 2     #12
[150, 128, 0],  # Truck 3     #13

# Small Vehicles Colors
[0, 255, 0],  # Small vehicles 1    
[0, 200, 0],  # Small vehicles 2    #15
[0, 150, 0],  # Small vehicles 3    #16
#--
# Traffic Signal Colors
[0, 128, 255],  # Traffic signal 1
[30, 28, 158],  # Traffic signal 2  #18
[60, 28, 100],  # Traffic signal 3  #19

# Traffic Sign Colors
[0, 255, 255],  # Traffic sign 1
[30, 220, 220],  # Traffic sign 2   #21
[60, 157, 199],  # Traffic sign 3   #22

# Utility Vehicle Colors
[255, 255, 0],  # Utility vehicle 1
[255, 255, 200],  # Utility vehicle 2  #24

# Other Colors
[233, 100, 0],  # Sidebars #25 
[110, 110, 0],  # Speed bumper
[128, 128, 0],  # Curbstone
[255, 193, 37],  # Solid line
[64, 0, 64],  # Irrelevant signs
[185, 122, 87],  # Road blocks
[0, 0, 100],  # Tractor
[139, 99, 108],  # Non-drivable street
[210, 50, 115],  # Zebra crossing
[255, 0, 128],  # Obstacles / trash
[255, 246, 143],  # Poles
[150, 0, 150],  # RD restricted area
[204, 255, 153],  # Animals
[238, 162, 173],  # Grid structure
[33, 44, 177],  # Signal corpus
[180, 50, 180],  # Drivable cobblestone
[255, 70, 185],  # Electronic traffic
[238, 233, 191],  # Slow drive area
[147, 253, 194],  # Nature object
[150, 150, 200],  # Parking area
[180, 150, 200],  # Sidewalk
[72, 209, 204],  # Ego car
[200, 125, 210],  # Painted driv. instr.
[159, 121, 238],  # Traffic guide obj.
[128, 0, 255],  # Dashed line
[255, 0, 255],  # RD normal street
[135, 206, 255],  # Sky
[241, 230, 255],  # Buildings
[96, 69, 143],  # Blurred area
[53, 46, 82]  # Rain dirt   #54  #42  

])

def process_image(path):
    img = cv2.imread(path)
    out_path = path.replace('/label/', '/multi_label/').replace('.png','.npy')
    # if os.path.exists(out_path):
        # return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    # img = cv2.resize(img, (512, 512))
    # out = np.zeros((img.shape[0], img.shape[1]))
    out = np.zeros((43,img.shape[0], img.shape[1]), np.uint8)

    
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            distances = cdist([img[i, j]], colors)
            closest_color_index = np.argmin(distances)
            closest_color_index = reducer(closest_color_index)
            out[closest_color_index, i, j] = 1

    print(out.shape)

    # print(path)
    # exit()

    # cv2.imwrite(out_path, out)
    np.save(out_path,out)
    print(out_path," : Success")
    
def reducer(index):
    if index in [0,1, 2, 3]: #3
        return 0 
    elif index in [4,5, 6, 7]: #3
        return 4  
    elif index in [8,9, 10]:  #2
        return 8
    elif index in [11,12, 13]: #2
        return 11  
    elif index in [14,15, 16]: #2
        return 14
    else:
        return index-12  



 

# process_image(A2D2_path_all_seg[0])
      
    
if __name__ == '__main__':
    with Pool(processes=48) as pool:  
        pool.map(process_image, A2D2_path_all_seg)
# {
#     "#ff0000": "Car 1" : (255, 0, 0),
#     "#c80000": "Car 2" : (200, 0, 0),
#     "#960000": "Car 3" : (150, 0, 0),
#     "#800000": "Car 4" : (128, 0, 0),
#     "#b65906": "Bicycle 1" : (182, 89, 6),
#     "#963204": "Bicycle 2" : (150, 50, 4),
#     "#5a1e01": "Bicycle 3" : (90, 30, 1),
#     "#5a1e1e": "Bicycle 4" : (90, 30, 30),
#     "#cc99ff": "Pedestrian 1" : (204, 153, 255),
#     "#bd499b": "Pedestrian 2" : (189, 73, 155),
#     "#ef59bf": "Pedestrian 3" : (239, 89, 191),
#     "#ff8000": "Truck 1" : (255, 128, 0),
#     "#c88000": "Truck 2" : (200, 128, 0),
#     "#968000": "Truck 3" : (150, 128, 0),
#     "#00ff00": "Small vehicles 1" : (0, 255, 0),
#     "#00c800": "Small vehicles 2" : (0, 200, 0),
#     "#009600": "Small vehicles 3" : (0, 150, 0),
#     "#0080ff": "Traffic signal 1" : (0, 128, 255),
#     "#1e1c9e": "Traffic signal 2" : (30, 28, 158),
#     "#3c1c64": "Traffic signal 3" : (60, 28, 100),
#     "#00ffff": "Traffic sign 1" : (0, 255, 255),
#     "#1edcdc": "Traffic sign 2" : (30, 220, 220),
#     "#3c9dc7": "Traffic sign 3" : (60, 157, 199),
#     "#ffff00": "Utility vehicle 1" : (255, 255, 0),
#     "#ffffc8": "Utility vehicle 2" : (255, 255, 200),
#     "#e96400": "Sidebars" : (233, 100, 0),
#     "#6e6e00": "Speed bumper" : (110, 110, 0),
#     "#808000": "Curbstone" : (128, 128, 0),
#     "#ffc125": "Solid line" : (255, 193, 37),
#     "#400040": "Irrelevant signs" : (64, 0, 64),
#     "#b97a57": "Road blocks" : (185, 122, 87),
#     "#000064": "Tractor" : (0, 0, 100),
#     "#8b636c": "Non-drivable street" : (139, 99, 108),
#     "#d23273": "Zebra crossing" : (210, 50, 115),
#     "#ff0080": "Obstacles / trash" : (255, 0, 128),
#     "#fff68f": "Poles" : (255, 246, 143),
#     "#960096": "RD restricted area" : (150, 0, 150),
#     "#ccff99": "Animals" : (204, 255, 153),
#     "#eea2ad": "Grid structure" : (238, 162, 173),
#     "#212cb1": "Signal corpus" : (33, 44, 177),
#     "#b432b4": "Drivable cobblestone" : (180, 50, 180),
#     "#ff46b9": "Electronic traffic" : (255, 70, 185),
#     "#eee9bf": "Slow drive area" : (238, 233, 191),
#     "#93fdc2": "Nature object" : (147, 253, 194),
#     "#9696c8": "Parking area" : (150, 150, 200),
#     "#b496c8": "Sidewalk" : (180, 150, 200),
#     "#48d1cc": "Ego car" : (72, 209, 204),
#     "#c87dd2": "Painted driv. instr." : (200, 125, 210),
#     "#9f79ee": "Traffic guide obj." : (159, 121, 238),
#     "#8000ff": "Dashed line" : (128, 0, 255),
#     "#ff00ff": "RD normal street" : (255, 0, 255),
#     "#87ceff": "Sky" : (135, 206, 255),
#     "#f1e6ff": "Buildings" : (241, 230, 255),
#     "#60458f": "Blurred area" : (96, 69, 143),
#     "#352e52": "Rain dirt" : (53, 46, 82)
# }
