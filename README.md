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
The script download data from the official source organize it inside Datasets/ folder 
```
sh get_data.sh
```


