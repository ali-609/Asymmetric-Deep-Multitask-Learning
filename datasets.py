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


class A2D2_symmetric(Dataset):
    def __init__(self,path):
        self.image_paths = path
        self.segmentation_dt=A2D2_seg(path)
        self.steering_dt=A2D2_steering(path)
        self.depth_dt=A2D2_depth(path)
        self.box_dt=A2D2_box(path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image=self.segmentation_dt[index]['image']

        segmentation=self.segmentation_dt[index]['A2D2_seg']

        steering=self.steering_dt[index]['A2D2_steering']

        depth=self.depth_dt[index]['A2D2_depth']

        boxes=self.box_dt[index]['A2D2_box']


        return {'dt_label':'A2D2_symmetric','image':image, 
                'A2D2_seg': segmentation, 'A2D2_steering': steering,'A2D2_depth': depth,'A2D2_box': boxes }

    

class A2D2_steering(Dataset):#standarilization 
    def __init__(self, path):
        self.image_paths = path
        # self.transforms = transforms.Compose([transforms.Resize((224 , 224)),
        self.transforms = transforms.Compose([transforms.Resize((1024, 1024)),
                                             transforms.ToTensor()])


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path=self.image_paths[index]

        json_path=img_path.replace('/camera_lidar_semantic/', '/bus/').replace('/camera/cam_front_center/','/bus/').replace('png','json')

        if not os.path.exists(json_path):
            steering_angles = np.zeros(21, dtype=np.float32)
        else:
            with open(json_path) as json_file:
                json_data = json.load(json_file)
                steering_angles = json_data.get('steering_angles')
        
                if steering_angles is None or not steering_angles:
                    steering_angles = np.zeros(21, dtype=np.float32)

            json_file.close()
        image = Image.open(img_path).convert('RGB') 
        image=self.transforms(image)


        steering=torch.tensor(steering_angles[-10:], dtype=torch.float32)
        steering/=70
        
        return {'dt_label':'A2D2_steering','image':image, 'A2D2_steering': steering }



class A2D2_seg(Dataset): #21 steering angle and segmentation
    def __init__(self, path):
       self.image_paths = path
    #    self.transforms = transforms.Compose([transforms.Resize((224, 224)),
       self.transforms = transforms.Compose([transforms.Resize((1024, 1024)),
                                             transforms.ToTensor()])
       self.transforms_seg = transforms.Compose([transforms.ToTensor()
                                            ])
       self.transforms_augment = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
   
])
   
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path=self.image_paths[index]
        seg_path=img_path.replace('/camera/', '/multi_label/').replace('_camera_', '_label_').replace('.png','.npy')
        
        image = Image.open(img_path).convert('RGB') 
        image=self.transforms(image)

        segmentation = np.load(seg_path).astype(np.float32)
        

        return {'dt_label':'A2D2_seg','image':image, 'A2D2_seg': segmentation }
    

       


class A2D2_box(Dataset):
    def __init__(self, path):
        # grid x grid x ()
        self.image_paths = path
        self.transforms = transforms.Compose([transforms.Resize((1024, 1024)),
                                             transforms.ToTensor()])
        self.width_scale=256/1920
        self.height_scale=256/1208

        # self.grid
        self.object_classes=['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path=self.image_paths[index]

        image = Image.open(img_path)
        image=self.transforms(image)

        json_path=img_path.replace('/camera/', '/label2D/').replace('_camera_','_label3D_').replace('png','json')
        box=np.zeros((190,8, 8))
        grid_size_x=(1920/box.shape[1])
        grid_size_y=(1208/box.shape[2])

        box_for_per_grid=np.zeros((8, 8),dtype=int)

        with open(json_path) as json_file:
            data = json.load(json_file)
            for i, items in enumerate(data):
                class_name=data[items]['class']
                class_index = self.object_classes.index(class_name)
    
                coordinates=data[items]['2d_bbox']
    
    
                box_center_x=(coordinates[0]+coordinates[2])/2
                box_center_y=(coordinates[1]+coordinates[3])/2
    
                box_width=coordinates[2]-coordinates[0]
                box_height=coordinates[3]-coordinates[1]
    
                grid_x=int(box_center_x//grid_size_x) #240)
                grid_y=int(box_center_y//grid_size_y) #151)
                
    
                if grid_x>7:
                    # print('before',box_width)
                    box_width=box_width-(1920/box.shape[1])*(abs(grid_x-7)*2)
                    # print('after',box_width,'-',abs(grid_x-7),'==',grid_x)
    
    
                if grid_x<0:
                    # print('before',box_width)
                    box_width=box_width-(1920/box.shape[1])*(abs(grid_x-0)*2)
                    # print('after',box_width,'-',abs(grid_x-0),'==',grid_x)
    
                ingrid_x=(box_center_x%grid_size_x)/grid_size_x
                ingrid_y=(box_center_y%grid_size_y)/grid_size_y
    
    
                scaled_box_width =  box_width/grid_size_x
                scaled_box_height = box_height/grid_size_y
    
    
    
                grid_x = np.clip(grid_x, 0, 7)
                grid_y = np.clip(grid_y, 0, 7)
                
                
    
                box[box_for_per_grid[grid_x,grid_y]*19,    grid_x, grid_y ]=1 # confidence
                box[box_for_per_grid[grid_x,grid_y]*19 + 1,grid_x, grid_y] = ingrid_x  # x-coordinate within the grid
                box[box_for_per_grid[grid_x,grid_y]*19 + 2,grid_x, grid_y] = ingrid_y  # y-coordinate within the grid
                box[box_for_per_grid[grid_x,grid_y]*19 + 3,grid_x, grid_y] = scaled_box_width  # width of box
                box[box_for_per_grid[grid_x,grid_y]*19 + 4,grid_x, grid_y] = scaled_box_height # height of box
                box[box_for_per_grid[grid_x,grid_y]*19 + 5 + class_index,grid_x, grid_y ] = 1 # class encoding
    
                box_for_per_grid[grid_x, grid_y]=box_for_per_grid[grid_x, grid_y]+1 
            
    
            # Convert 3D points to a PyTorch tensor
            json_file.close()
            boxes = torch.tensor(box, dtype=torch.float32)

        return {'dt_label':'A2D2_box','image':image,'A2D2_box': boxes}
    

class A2D2_depth(Dataset):
    def __init__(self, path):
        # grid x grid x ()
        self.image_paths = path
        self.transforms = transforms.Compose([transforms.Resize((1024, 1024)),
                                             transforms.ToTensor(),
                                            #  transforms.RandomResizedCrop(size=(256, 256)),
                                            #  transforms.RandomAffine(degrees=(0, 0),translate=(0.1, 0.3), scale=(0.5, 0.9))
                                             ]) # Shift My opinion: Transformation


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        img_path=self.image_paths[index]
        
        image = Image.open(img_path)
        image=self.transforms(image)
        depth_path=img_path.replace('/camera/', '/depth/').replace('_camera_','_depth_')

        depth=Image.open(depth_path)
        depth=self.transforms(depth)
        # depth=self.map_lidar_points_onto_image(image_orig=image,lidar=lidar)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        return {'dt_label':'A2D2_depth','image':image,'A2D2_depth': depth}
    


def a2d2_dataloader(base_path="./Datasets/camera_lidar_semantic_bboxes/", train_split=0.8, val_split=0.2):
    insult_list=['./Datasets/camera_lidar_semantic_bboxes/20180810_142822/camera/cam_front_center/20180810142822_camera_frontcenter_000000004.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000020.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000022.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000041.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000049.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000050.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000055.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000057.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132300/camera/cam_front_center/20181107132300_camera_frontcenter_000000063.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000035.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000051.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000055.png',
             './Datasets/camera_lidar_semantic_bboxes/20181107_132730/camera/cam_front_center/20181107132730_camera_frontcenter_000000061.png',
             './Datasets/camera_lidar_semantic_bboxes/20181108_103155/camera/cam_front_center/20181108103155_camera_frontcenter_000000028.png',
             './Datasets/camera_lidar_semantic_bboxes/20181108_103155/camera/cam_front_center/20181108103155_camera_frontcenter_000000033.png'
             ]

    insult_list=[os.path.abspath(path) for path in insult_list]



    all_folders = sorted(os.listdir(base_path))
    all_folders = all_folders[0:-3]  # Adjust according to your specific needs
    
    A2D2_path_train = [] 
    A2D2_path_val = []

    for folder in all_folders:
        folder_path = os.path.join(base_path, folder)
        
        # Get a list of all files in the current folder
        files_in_folder = np.array(sorted(glob.glob(os.path.join(folder_path, "camera/cam_front_center/*.png"))))
        files_in_folder = files_in_folder[~np.isin(files_in_folder, insult_list)]
        
        # Calculate the split indices
        split_index = int(len(files_in_folder) * train_split)
        
        # Split the data into training and validation sets for the current folder
        train_set = files_in_folder[:split_index]
        val_set = files_in_folder[split_index:]
        
        # Accumulate the sets for each folder
        A2D2_path_train.extend(train_set)
        A2D2_path_val.extend(val_set)
    
    return A2D2_path_train, A2D2_path_val

       

