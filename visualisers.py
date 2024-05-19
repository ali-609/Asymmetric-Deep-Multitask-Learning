import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


import torchvision.transforms.functional as TF


def process_array_multi_reduced(img):
    colors=np.array([
    # Car Colors
[255, 0, 0],  # Car 1
[200, 0, 0],  # Car 2
[150, 0, 0],  # Car 3
[128, 0, 0],  # Car 4

# Bicycle Colors
[182, 89, 6],  # Bicycle 1
[150, 50, 4],  # Bicycle 2
[90, 30, 1],  # Bicycle 3
[90, 30, 30],  # Bicycle 4

# Pedestrian Colors
[204, 153, 255],  # Pedestrian 1
[189, 73, 155],  # Pedestrian 2
[239, 89, 191],  # Pedestrian 3

# Truck Colors
[255, 128, 0],  # Truck 1
[200, 128, 0],  # Truck 2
[150, 128, 0],  # Truck 3

# Small Vehicles Colors
[0, 255, 0],  # Small vehicles 1
[0, 200, 0],  # Small vehicles 2
[0, 150, 0],  # Small vehicles 3

# Traffic Signal Colors
[0, 128, 255],  # Traffic signal 1
[30, 28, 158],  # Traffic signal 2
[60, 28, 100],  # Traffic signal 3

# Traffic Sign Colors
[0, 255, 255],  # Traffic sign 1
[30, 220, 220],  # Traffic sign 2
[60, 157, 199],  # Traffic sign 3

# Utility Vehicle Colors
[255, 255, 0],  # Utility vehicle 1
[255, 255, 200],  # Utility vehicle 2

# Other Colors
[233, 100, 0],  # Sidebars
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
[53, 46, 82]  # Rain dirt

])
    out = np.zeros((img.shape[1], img.shape[2],3), np.uint8)

    # for i in range(img.shape[1]):
    #     for j in range(img.shape[2]):
    index=np.argmax(img,axis=0)

    index[index > 16] += 12


    out[:,:,:]=colors[index]
    print(out.shape)
        
    return out

def visualize_seg(seg_output,file_name):
    output=process_array_multi_reduced(seg_output)
    output_image=Image.fromarray(output, 'RGB')
    output_image.save(file_name)

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

def visualize_box(image,boxes,file_name,threshold=1.0):
    image_np = np.array(TF.to_pil_image(image))
    dataset=['Car', 'Pedestrian', 'Truck', 'VanSUV', 'Cyclist', 'Bus', 'MotorBiker', 'Bicycle', 'UtilityVehicle', 'Motorcycle', 'CaravanTransporter', 'Animal', 'Trailer', 'EmergencyVehicle']


    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    # Iterate over each grid cell and draw bounding boxes
    for grid_x in range(boxes.shape[1]):
        for grid_y in range(boxes.shape[2]):
            for nth_anchor in range(10):
            # Confidence score
                confidence = boxes[nth_anchor*19, grid_x, grid_y]
                # If confidence score is above a threshold, draw the bounding box
                if confidence >= threshold:
                    if confidence>0.8929 and confidence<0.8931:
                        continue  # Adjust threshold as needed
                    print(confidence)
                    # Bounding box coordinates
                    x =  grid_x * (image_np.shape[1]/8)+ boxes[nth_anchor*19+1, grid_x, grid_y] * (image_np.shape[1]/8)  # x-coordinate within the grid
                    y =  grid_y * (image_np.shape[0]/8)+ boxes[nth_anchor*19+2, grid_x, grid_y] * (image_np.shape[0]/8)  # y-coordinate within the grid
                    width = boxes[nth_anchor*19+3, grid_x, grid_y] * (image_np.shape[1] /8) # Width of the box
                    height = boxes[nth_anchor*19+4, grid_x, grid_y] * (image_np.shape[0]/8)  # Height of the box
                    # Convert center coordinates to top-left coordinates
                    x1 = int(x - width / 2)
                    y1 = int(y - height / 2)
                    x2 = int(x + width / 2)
                    y2 = int(y + height / 2)
                    # Draw the bounding box
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                                           fill=False, edgecolor='r', linewidth=2))
               
                    # Get the class index
                    class_index = np.argmax(boxes[nth_anchor*19+5:nth_anchor*19+14, grid_x, grid_y])
                    class_name = dataset[class_index]
                    
                        # Show class label
                    # plt.text(x1, y1 - 5, class_name, color='r', fontsize=10, ha='left', va='center')
    
    plt.savefig(file_name)
    plt.close()
