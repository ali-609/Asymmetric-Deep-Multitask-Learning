import sys
from pathlib import Path

import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--threads', type=int, default=4)

args = parser.parse_args()
num_threads = args.threads

rel_dirs=sorted(glob.glob("./Datasets/camera_lidar_semantic_bboxes/2018*/label/cam_front_center/*.png"))


A2D2_path_all_seg = [os.path.abspath(path) for path in rel_dirs]

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
    if os.path.exists(out_path):
        print(out_path," : Exist")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))

    out = np.zeros((43,img.shape[0], img.shape[1]), np.uint8)

    
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):


            closest_color_index = np.argmin(np.sum((colors - img[i, j])**2, axis=1))
            closest_color_index = reducer(closest_color_index)
            out[closest_color_index, i, j] = 1





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


print('CPU Threads in use: ', num_threads)
      
    
if __name__ == '__main__':
    with Pool(processes=num_threads) as pool:  
        pool.map(process_image, A2D2_path_all_seg)
