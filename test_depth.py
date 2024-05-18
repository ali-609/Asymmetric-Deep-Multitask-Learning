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
from datasets import A2D2_steering,A2D2_seg,A2D2_box
from losses import YOLOLoss 
from models.yolo_v1 import YOLOv1
from models.resnet_bifpn import ResNETBiFPN
from models.UNet_depth import UNet
from datasets import A2D2_steering,A2D2_seg,A2D2_depth
import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.DenseDepth import PTModel
from models.steering_onlyUNet import HydraUNetSTR
# from UNet import UNet
import os
from datetime import datetime
import wandb
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
# from models.resnet_bifpn import ResNETBiFPN
from models.resnet_bifpn_depth import ResNETBiFPN




TRAIN_SPLIT = 0.8



# BASE_PATH = "/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/"
BASE_PATH="/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/" 


all_folders = sorted(os.listdir(BASE_PATH))
all_folders = all_folders[1:-3]
print(all_folders)


A2D2_path_train_seg=[]#A2D2_path_all[:int(len(A2D2_path_all) * TRAIN_SPLIT)]
A2D2_path_train_str=[]

A2D2_path_val_seg = []
A2D2_path_val_str = []  #A2D2_path_all[-int(len(A2D2_path_all) * VAL_SPLIT):]

A2D2_path_train  = [] 
A2D2_path_val = []

for folder in all_folders:
    folder_path = os.path.join(BASE_PATH, folder)
    
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




A2D2_dataset_train_dep=A2D2_depth(A2D2_path_train)
A2D2_dataset_val_dep=A2D2_depth(A2D2_path_val)
# print(A2D2_dataset_train_box[0])

reverse_transform = transforms.Compose([
    transforms.ToPILImage(),

])

def visualize_depth(output, file_name):
    depth_array = output.squeeze()

    cmap = plt.cm.inferno
    plt.figure(figsize=(10.24, 10.24))
    # Plot the depth array with the chosen colormap
    plt.imshow(depth_array, cmap=cmap)
    # plt.colorbar()  # Add a color bar for reference

    # plt.title('Depth Visualization')
    plt.xticks([])
    plt.yticks([])
    plt.gcf().patch.set_facecolor('none')
    plt.gca().patch.set_facecolor('none')

    # Save the plot as an image file with the provided file name
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=130)  # dpi=100 for 1024x1024 resolution

    # Close the plot to prevent displaying it
    plt.close()



sample_no=245
sample=A2D2_dataset_val_dep[sample_no]


directory = f"results/depth/{sample_no}"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)


print('Truth: ')

real_image=sample['image']
real_image=reverse_transform(real_image)
real_image.save(f"results/depth/{sample_no}/_real.png")

visualize_depth(sample['A2D2_depth'], f"results/depth/{sample_no}/_truth.png")


# exit()
print("Asymmetric: ")
model= ResNETBiFPN()
model.load_state_dict(torch.load('weights/universal_asymmetric_2024-05-07_20-56-31.pth' )) 
# model.to(device=device)
model.eval()



sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

# print(output)
depth_array=output[3][0]
visualize_depth(depth_array, f"results/depth/{sample_no}/_asymmetric.png")


print("Symmetric: ")
model= ResNETBiFPN()
model.load_state_dict(torch.load('weights/universal_symmetric_2024-05-02_18-34-33.pth' ))
model.eval()



sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

# print(output)
depth_array=output[3][0]
visualize_depth(depth_array, f"results/depth/{sample_no}/_symmetric.png")

print("Universal: ")
model= ResNETBiFPN()
model.load_state_dict(torch.load('weights/universal_depth_2024-05-01_10-41-17.pth' ))
model.eval()



sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

# print(output)
depth_array=output[3][0]
visualize_depth(depth_array, f"results/depth/{sample_no}/_universal.png")

print("Single task: ")
model = PTModel()
model.load_state_dict(torch.load('weights/depth_2024-04-24_01-22-43.pth' ))
model.eval()



sample_input=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(sample_input)

# print(output)
depth_array=output[0]
visualize_depth(depth_array, f"results/depth/{sample_no}/_single_task.png")

exit()


# depth_array=sample['A2D2_depth'].squeeze()

cmap = plt.cm.inferno
plt.figure(figsize=(10.24, 10.24))
# Plot the depth array with the chosen colormap
plt.imshow(depth_array, cmap=cmap)
# plt.colorbar()  # Add a color bar for reference

# plt.title('Depth Visualization')
plt.xticks([])
plt.yticks([])
plt.gcf().patch.set_facecolor('none')
plt.gca().patch.set_facecolor('none')

# Save the plot as an image file
plt.savefig('depth_visualization.png', bbox_inches='tight', pad_inches=0, dpi=100)  # dpi=100 for 1024x1024 resolution

# Close the plot to prevent displaying it
plt.close()
exit()

image=sample['image']
depth_ground_truth=sample['A2D2_depth']
depth_prediction=output[3][0]
print(depth_prediction)

reverse_transform = transforms.Compose([
    transforms.ToPILImage(),

])

from PIL import Image
#*15
# depth_prediction=depth_prediction*2
depth_prediction=reverse_transform(depth_prediction)
depth_ground_truth=reverse_transform(depth_ground_truth)
depth_image=reverse_transform(image)

ground_truth_file = f"depth_truth_big.png"
prediction_file = f"depth_prediction_big.png"
depth_image_file= f"depth_image_big.png"
# Save the RGB images

depth_prediction.save(prediction_file)
depth_ground_truth.save(ground_truth_file)
depth_image.save(depth_image_file)
# exit()

depth_ground_truth = np.array(depth_ground_truth)
data=depth_ground_truth
print("Mean: ",depth_ground_truth.mean())
print("Max: ",depth_ground_truth.max())
print("Min: ",depth_ground_truth.min())
# depth_ground_truth=cv2.fromarray(depth_ground_truth)
# depth_ground_truth = cv2.cvtColor(depth_ground_truth,cv2.COLOR_GRAY2RGB)

print("SHAPE: ",depth_ground_truth.shape)

def save_depth_community(data,name):
   data_without_zeros = np.where(data == 0, np.nan, data)
   
   # Normalize the data
   normalized_data = (data_without_zeros - 0) / (depth_ground_truth.max() - 0)
   
   # Choose a colormap, for example 'viridis'
   cmap = plt.cm.jet
   
   # Create the color-mapped image with zero values masked
   color_mapped_image = cmap(np.ma.masked_invalid(normalized_data))
   
   plt.axis('off')  # Turn off axis
   
   plt.imshow(color_mapped_image)  # Display the image
   
   plt.savefig(name, bbox_inches='tight', pad_inches=0)

# save_depth_community(depth_ground_truth,"depth_truth_colored.png")
# save_depth_community(depth_prediction,"depth_prediction_colored.png")
normalized_data = data / 255

# # Choose a colormap, for example 'viridis'
cmap = plt.cm.jet

# # Create the color-mapped image
color_mapped_image = cmap(data)


# plt.axis('off')  # Turn off axis


# plt.savefig('depth_truth_colored.png', bbox_inches='tight', pad_inches=0)

# cv2.imwrite('depth_truth_colored.png',depth_ground_truth)
# depth_map_np[depth_map_np == 0] = np.nan
# Normalize the depth values to the range [0, 1]
# normalized_depth_map = cv2.normalize(depth_map_np, None, 0, 1, cv2.NORM_MINMAX)

# Apply a colormap
# colormap_depth = cv2.applyColorMap((normalized_depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Convert NumPy array back to PIL image
# colormap_depth_pil = Image.fromarray(cv2.cvtColor(colormap_depth, cv2.COLOR_BGR2RGB))

# Display the result (optional)
# colormap_depth_pil.show()

# Save the result
# colormap_depth_pil.save('colormap_depth_map.png')