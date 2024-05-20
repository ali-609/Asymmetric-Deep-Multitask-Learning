# Asymmetric-Deep-Multitask-Learning
This is the code repository of the thesis "Asymmetric Deep Multitask Learning." The repo has tools to automate the downloading, extracting, and processing of the A2D2 database for four tasks. In addition, the repo provides code to train several models and visualization tools.

# System requirements:
x86 CPU with at least four core <br>
Nvidia GPU with 12GB VRAM <br>
16GB RAM <br>
Conda 24.1.0 <br>
The codebase was developed on HPC, and some processes may cause overhead in consumer laptops or PCs. If feasible, please run the code on either an HPC or a high-performance workstation.

# Conda environment
Use the given yaml file to create a conda environmet:
```
conda env create -f ADMTL_environment.yml
conda activate ADMTL_environment
```

# Data
## Preparing data
The script downloads data from the official source and organizes it inside Datasets/ folder. 
```
sh get_data.sh
```

## Preprocessing data
All codes for preprocessing are located in the 'preprocessing' folder. Run all codes given from the project's root directory. 

### Segmentation labels
The code takes RGB segmentation labels, extracts color annotations from the image and stores the class information in .npy files. Preprocessing segmentation labels is the most resource-demanding preprocessing process. 
```
python preprocessing/segmentation.py --threads 4
```
Code optional '--threads' argument defines the number of CPU cores for the process. By default, the code uses four CPUs. The code is able to check if the given label is already processed or not, which means after it is stopped, the code continues to preprocess labels where it left.

### Depth estimation labels
Preprocessing depth estimation involves taking the depth information of pixels from the lidar and applying that information to an empty grayscale image.
```
python preprocessing/depth_preprocess.py --threads 4
```
This code also uses several CPU cores parallelly, but preprocessing depth labels takes much less time than the segmentation one. With sufficient CPU cores, preprocessing all depth labels takes 10-15 minutes.

### Bounding Box labels


