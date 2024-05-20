# Asymmetric-Deep-Multitask-Learning
This is the code repository of the thesis "Asymmetric Deep Multitask Learning." The repo has tools to automate the downloading, extracting, and processing of the A2D2 database for four tasks. In addition, the repo provides code to train several models and visualization tools.

# System requirements:
x86 CPU with at least 8 core <br>
CUDA capable Nvidia GPU with 12GB VRAM <br>
32GB RAM <br>
conda: 24.1.0 <br>
glibc: 2.17 <br>
1 TB storage <br>
The codebase was developed on HPC, and some processes may cause overhead in consumer laptops or PCs. If feasible, please run the code on either an HPC or a high-performance workstation.

# Setting environment
Use the given yml file to create a conda environment:
```
conda env create -f ADMTL_environment.yml
conda activate admtl_env
```

In case yml file fail, try to install mentioned libraries manually:
numpy, pillow, opencv-python, torch, torchvision, argparse, tqdm, wandb

To initialize wandb, please refer 'https://docs.wandb.ai/quickstart'


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
This code cleans duplicate 2D bounding boxes and stores them in separate files.
```
python preprocessing/bounding_box.py --threads 4
```

### Steering angle labels
The below code takes bus gateway information of every frame from one big .json file and stores them in smaller files.
```
 python preprocessing/steering_angle.py 
```
Unlike the preprocessing of other labels for steering angle labels, no parallel CPU core is used as code takes about a minute in one CPU core to preprocess all labels.


# Train
Some models have too many parameters, making using command-line arguments a bit inconvenient. For that reason, train parameters like learning rate or batch size are stored in .yaml files in 'configs/' directory.
The table below gives information about which code train, what kind of model, and the associated .yaml file.

| Program file  | YAML file |  Possible train configurations |
| ------------- | ------------- |------------- |
| train_single_task.py  | configs/single_task_conf.yaml  | PilotNet for Steering angle prediction; UNet for semantic segmentation, DenseDepth for depth estimation, YOLOv1  for bounding box estimation |
| train_symmetric.py  | configs/symmetric_conf.yaml  | MTL with symmetric labels, MTL with one label presenting among 4 present  |
| train_asymmetric.py  |  configs/asymmetric_conf.yaml | Asymmetric MTL  |


It is highly recommended to check the YAML file of the training program before any training process.


Example YAML file:
```
batch_sizes:  accumulate steps for decoders and backbone
  batch_backbone: 48
  batch_steering: 24
  batch_segmentation: 32
  batch_box: 16
  batch_depth: 16

learning_rates: # learning rates optimizers of decoders and backbone
  backbone_lr: 0.002
  segmentation_lr: 0.0002
  steering_lr: 0.00004
  box_lr: 0.00001
  depth_lr: 0.00003

coefficients:  # coefficients to balance loss functions
  steering_coef: 1
  segmentation_coef: 80
  box_coef: 400
  depth_coef: 200
 
n_epochs: 10 # number of train epoch 

num_workers: 8 #number of CPU cores

```

### Single task train
Single-task training includes training specialized single-task models on single-label data.
```
python train_single_task.py --train-mode ['segmentation', 'steering', 'box', 'depth']
```
'--train-mode' takes one of four given arguments and can't take multiple arguments. If no argument is given, the program train not run.


### Multi task train
The program can train MTL architecture in five configurations: Symmetric labels, where all labels are presented, or one of four possible labels.
```
python train_symmetric.py --train-mode choices=['segmentation', 'steering', 'box', 'depth', 'symmetric'], default='symmetric'
```
'--train-mode' takes one of one of five possible arguments. If no arguments are given, the program will use a symmetric configuration.

### Asymmetric Multi task train
The below program train MTL on asymmetric labels:
```
python train_asymmetric.py
```

# Evaluation
The script mentioned below is to obtain pre-trained weights that were used during research:
```
sh get_weight.sh
```
These weights are also listed in the 'configs/default_weights_conf.yaml' file, which defines default weights if the user doesn't enter the weight file during evaluation.
```
# configs/default_weights_conf.yaml
asymmetric_weight: 'weights/universal_asymmetric_2024-05-07_20-56-31.pth'

symmetric_weight: 'weights/universal_symmetric_2024-05-02_18-34-33.pth'

single_task_weight:
  segmentation: 'weights/segmentation_2024-04-21_00-16-07.pth' 
  boundingbox: 'weights/yolo_2024-05-03_19-36-16.pth'
  depth: 'weights/depth_2024-04-24_01-22-43.pth'
  steering: 'weights/steering_2024-04-25_04-27-04.pth'

universal_task_weight:
  segmentation: 'weights/universal_segmentation_2024-05-01_10-42-53.pth'
  boundingbox: 'weights/universal_box_2024-05-02_18-42-46.pth'
  depth: 'weights/universal_depth_2024-05-01_10-41-17.pth'
  steering: 'weights/universal_steering_2024-05-02_18-44-59.pth'
```

It is recommended to download weights by script or train and obtain weights files for all possible cases in 'configs/default_weights_conf.yaml'.
After training, new weights need to be entered into 'configs/default_weights_conf.yaml'


## Metric calculation
The below program is used to calculate metrics for given configuration:
```
python metrics.py --data ['segmentation', 'steering', 'boundingbox', 'depth'] --model ['PilotNet', 'UNet', 'YOLO', 'DenseDepth', 'MTL'] --variant ['symmetric', 'asymmetric','single-task'] --weights /path/to/weight/file
```

Among arguments, '--data' and '--model' are required, and the '--weights' argument is optional in case the user doesn't use that argument; the program takes weights from 'configs/default_weights_conf.yaml'.
'--variant' argument is used when the 'MTL' option is used and '--weights' option is not used, this argument determines which version program should take among the default weights of the MTL architecture

## Visualisation
The below program visualizes the outputs of asymmetric-trained MTL, symmetric-trained MTL, single-label-trained MTL, and STL models and stores them in one folder.
```
python visualize.py --task ['segmentation','boundingbox', 'depth'] --sample range(0, 2486) --threshold (float)
```
The program stores all results with ground truth labels in the 'results/{task}/{sample}/' structured directory. Among arguments, '--task' and '--sample' are required arguments. The '--threshold' argument is only needed for the bounding box task.






