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
from datasets import A2D2_steering,A2D2_seg,A2D2_depth
from losses import YOLOLoss,DepthLoss

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
from models.UNet_depth import UNet
# from UNet import UNet
import os
from datetime import datetime
import wandb
#RSync



class AsymmetricTrainer(nn.Module):
    def __init__(self, optimizer, batch_size ,loss_func, device, wandb_log,data_len_tr,data_len_val,grad_dec_coef=1):
        super(AsymmetricTrainer, self).__init__()

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.device = device

        self.counter = 0
        # self.backbone_counter = 0 
        # self.task_train_loss=0
        self.batch_loss = 0
        self.grad_dec_coef=grad_dec_coef
        self.total_train_loss=0
        self.total_validation_loss=0
        # self.common_batch_loss=0

        self.wandb_train_log = "Training " + wandb_log + " Loss"
        self.wandb_average_train_log = "Average Train " + wandb_log + " Loss" 
        self.wandb_average_validation_log = "Average Validation " + wandb_log + " Loss"
        self.data_len_tr=data_len_tr
        self.data_len_val=data_len_val

        if wandb_log == 'Segmentation' or wandb_log == 'Steering':
            self.wandb_train_log = self.wandb_train_log + ' '  # Adding empty space at the end



    def train_batch(self, model, data):

        self.counter += 1

        inputs = data["image"].to(device=self.device)
        label = data[data['dt_label'][0]].to(device=self.device)


        steering_output, segmentation_output, box_output,depth_output = model(inputs)
        output_dict={'A2D2_steering':steering_output,
                     'A2D2_seg':segmentation_output,
                     'A2D2_box':box_output,
                     'A2D2_depth':depth_output}



        loss_value=self.loss_func(output_dict[data['dt_label'][0]],label)



        self.total_train_loss += loss_value
        log_loss=loss_value/self.grad_dec_coef
        # self.task_train_loss += loss_value


        self.batch_loss += loss_value


        loss_value = (loss_value/self.batch_size)/self.grad_dec_coef



        loss_value.backward() #Backprob

        if self.counter % self.batch_size == 0:
            self.optimizer.step() #Apply
            self.optimizer.zero_grad()
            self.counter = 0

            wandb.log({self.wandb_train_log: self.batch_loss / self.batch_size})
            self.batch_loss = 0

        return model,log_loss
    

    def validate_batch (self, model, data):
        inputs = data["image"].to(device=self.device)
        label = data[data['dt_label'][0]].to(device=self.device)

        steering_output, segmentation_output, box_output,depth_output = model(inputs)
        output_dict={'A2D2_steering':steering_output,
                     'A2D2_seg':segmentation_output,
                     'A2D2_box':box_output,
                     'A2D2_depth':depth_output}


        loss_value=self.loss_func(output_dict[data['dt_label'][0]],label)

        self.total_validation_loss=self.total_validation_loss+loss_value

        log_loss=loss_value/self.grad_dec_coef


        return log_loss
    
    def end_epoch(self):
        avgTrainLoss=self.total_train_loss/self.data_len_tr
        avgValLoss=self.total_validation_loss/self.data_len_val
            

        wandb.log({
               self.wandb_average_train_log: avgTrainLoss,
               self.wandb_average_validation_log: avgValLoss,
    })

        self.total_train_loss=0
        self.total_validation_loss=0
    
    