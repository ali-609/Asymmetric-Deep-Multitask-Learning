import sys
from pathlib import Path
from tqdm import tqdm


import torch

from torch.utils.data import Dataset, random_split, DataLoader,ConcatDataset



from datasets import A2D2_steering,A2D2_seg,A2D2_depth,A2D2_box,a2d2_dataloader

import random
import torch
import torch.nn as nn




from models.UNet import UNet
from models.pilotnet import PilotNet
from models.yolo_v1 import YOLOv1
from models.DenseDepth import PTModel

import wandb
from datetime import datetime
import os
from losses import YOLOLoss,DepthLoss


import argparse

torch.manual_seed(42)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print('Time of run: ', current_time)

parser = argparse.ArgumentParser()
parser.add_argument('--train-mode', choices=['segmentation', 'steering', 'box', 'depth'])
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--n-epoch', type=int, default=10)


args = parser.parse_args()

train_mode = args.train_mode
BATCH_SIZE=args.batch_size
lr=args.learning_rate
num_workers = args.num_workers
n_epochs = args.n_epoch

print("Train mode: ",train_mode)

base_name = f"{train_mode}"
filename = f"weights/{base_name}_{current_time}.pth"

if not os.path.exists("weights/"):
    os.makedirs("weights/")



A2D2_path_train,A2D2_path_val=a2d2_dataloader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if train_mode == 'segmentation':
    A2D2_dataset_train = A2D2_seg(A2D2_path_train)
    A2D2_dataset_val = A2D2_seg(A2D2_path_val)

    model=UNet()
    loss=nn.CrossEntropyLoss()

    wandb_train="Training Segmentation Loss "
    wandb_avg_train="Average Train Segmentation Loss"
    wandb_avg_val="Average Validation Segmentation Loss"

elif train_mode == 'steering':
    A2D2_dataset_train = A2D2_steering(A2D2_path_train)
    A2D2_dataset_val = A2D2_steering(A2D2_path_val)

    model=PilotNet()
    loss=nn.L1Loss()

    wandb_train="Training Steering Loss "
    wandb_avg_train="Average Train Steering Loss"
    wandb_avg_val="Average Validation Steering Loss"
elif train_mode == 'box':
    A2D2_dataset_train = A2D2_box(A2D2_path_train)
    A2D2_dataset_val = A2D2_box(A2D2_path_val)

    model=YOLOv1()
    loss=YOLOLoss()

    wandb_train="Training Boxes Loss"
    wandb_avg_train="Average Train Boxes Loss"
    wandb_avg_val="Average Validation Boxes Loss"
elif train_mode == 'depth':
    A2D2_dataset_train = A2D2_depth(A2D2_path_train)
    A2D2_dataset_val = A2D2_depth(A2D2_path_val)

    model=PTModel()
    loss=DepthLoss()

    wandb_train="Training Depth Loss"
    wandb_avg_train="Average Train Depth Loss"
    wandb_avg_val="Average Validation Depth Loss"


train_dataloader = DataLoader(A2D2_dataset_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers)
val_dataloader = DataLoader(A2D2_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

config = {
    "learning_rate": lr,
    "batch": BATCH_SIZE
}

run_name = f"{base_name}_{current_time}"

wandb.init(project="thesis_final", config=config, name=run_name)
wandb.watch(model)


best_val_loss=float('inf')
for epoch in range(n_epochs):
    
    model.train()

    total_training_loss = 0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
        
        inputs = data["image"].to(device=device) 
        label = data[data['dt_label'][0]].to(device=device)


        #Output

        output = model(inputs)

        # Loss calculation
        loss_value = loss(output, label)

        #Backward
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Logging
        
        total_training_loss += loss_value


        wandb.log({wandb_train: loss_value})

    avgTrainLoss = total_training_loss / len(train_dataloader)


    model.eval()
    total_validation_loss = 0

    for i, data in enumerate( tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
        
        inputs = data["image"].to(device=device) 
        label = data[data['dt_label'][0]].to(device=device)


        #Output
        output = model(inputs)

        # Loss calculation
        loss_value = loss(output, label)

        total_validation_loss=total_validation_loss+loss_value
    
    avgValLoss = total_validation_loss / len(val_dataloader)

    wandb.log({
               wandb_avg_train: avgTrainLoss,
               wandb_avg_val: avgValLoss
    })

    if avgValLoss<best_val_loss:
        best_val_loss=avgValLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)



    





