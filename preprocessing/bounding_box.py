import json
import glob
import os
from multiprocessing import Pool

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--threads', type=int, default=4)

args = parser.parse_args()
num_threads = args.threads


rel_dirs=sorted(glob.glob('./Datasets/camera_lidar_semantic_bboxes/2018*/label3D/cam_front_center/*.json'))

files = [os.path.abspath(path) for path in rel_dirs]

object_classes = ['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']

def remove_dublicate(file):
    out_path = file.replace('/label3D/', '/label2D/')  # Change the output folder name
    output = {}

    with open(file) as filel:
        data = json.load(filel)

    unique_boxes = set()

    filtered_data = {}

    for i, items in enumerate(data):
        class_name = data[items]['class']
        coordinates = data[items]['2d_bbox']

        # Convert coordinates and class to a string for easy comparison
        box_key = f"{coordinates[0]}_{coordinates[1]}_{coordinates[2]}_{coordinates[3]}_{class_name}"

        if box_key not in unique_boxes:
            unique_boxes.add(box_key)


            # Add the unique box to the filtered data
            filtered_data[items] = {
                'class': class_name,
                '2d_bbox': coordinates
            }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save the filtered data to a new JSON file
    with open(out_path, 'w') as outfile:
        json.dump(filtered_data, outfile,indent=2)

    print(f"Filtered data saved to: {out_path}")

print('CPU Threads in use: ', num_threads)
      
    
if __name__ == '__main__':
    with Pool(processes=num_threads) as pool:  
        pool.map(remove_dublicate, files)
