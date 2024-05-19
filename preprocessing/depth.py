import sys
from pathlib import Path

import cv2
import numpy as np
import os


import glob

from PIL import Image


from multiprocessing import Pool
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--threads', type=int, default=4)

args = parser.parse_args()
num_threads = args.threads

rel_dirs=sorted(glob.glob("./Datasets/camera_lidar_semantic_bboxes/*/camera/cam_front_center/*.png"))


BASE_PATH = [os.path.abspath(path) for path in rel_dirs]



def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=5, pixel_opacity=1):
    image_orig=np.copy(image_orig)
    image = np.zeros((image_orig.shape[0],image_orig.shape[1]), dtype=np.uint8)

    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int64)
    cols = (lidar['col'] + 0.5).astype(np.int64)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']  
    # determine point colours from distance
    colours = distances  / 123
    # colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        # np.sqrt(pixel_opacity), 1.0)) for c in colours])
    # colours = 
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)

        image[pixel_rows, pixel_cols] = 255 * colours[i]
    return image.astype(np.uint8)




# for img_path in tqdm(BASE_PATH):
   
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



def process_image(img_path):
    if img_path in insult_list:
        return 0
    out_path=img_path.replace('/camera/', '/depth/').replace('_camera_','_depth_')
    print(img_path)
    if os.path.exists(out_path):
        print(out_path," : Exist")
        return 0
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass
    
    image = Image.open(img_path)

    lidar_path=img_path.replace('/camera/', '/lidar/').replace('_camera_','_lidar_').replace('png','npz')
    lidar=np.load(lidar_path)

    depth = map_lidar_points_onto_image(image_orig=image, lidar=lidar)



    cv2.imwrite(out_path, depth)
    print(" : Success")



if __name__ == '__main__':
    with Pool(processes=num_threads) as pool:  
        pool.map(process_image, BASE_PATH)
    
