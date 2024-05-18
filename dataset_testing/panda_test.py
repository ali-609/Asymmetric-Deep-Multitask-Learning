import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models
from skimage.util import random_noise
import glob
from datasets import *
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
import json

import gzip
import pickle

# Specify the path to your .pkl.gz file
file_path = 'Datasets/Panda/001/annotations/semseg/00.pkl.gz'

# Open the file in binary read mode
with gzip.open(file_path, 'rb') as f:
    # Load the pickled object
    unpacked_object = pickle.load(f)

# Do something with the unpacked object
print(unpacked_object)