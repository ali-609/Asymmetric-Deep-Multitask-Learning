import json
import glob
import os

files = sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/camera_lidar_semantic_bboxes/2018*/label3D/cam_front_center/*.json"))
A2D2_path_all = sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))

print(len(files))
print(len(A2D2_path_all))

object_classes = ['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']

for file in files:
    out_path = file.replace('/label3D/', '/label2D/')  # Change the output folder name
    output = {}

    with open(file) as filel:
        data = json.load(filel)

    unique_boxes = set()
    unique_classes = set()  # To store unique boxes based on coordinates and class
    filtered_data = {}

    for i, items in enumerate(data):
        class_name = data[items]['class']
        coordinates = data[items]['2d_bbox']

        # Convert coordinates and class to a string for easy comparison
        box_key = f"{coordinates[0]}_{coordinates[1]}_{coordinates[2]}_{coordinates[3]}_{class_name}"

        if box_key not in unique_boxes:
            # if class_name not in unique_classes:
            unique_boxes.add(box_key)
                # unique_classes.add(class_name)


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
