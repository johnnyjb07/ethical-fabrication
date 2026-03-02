from util import data_generate, train, cnnlstm, cnn3d, DPtrain

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
batch_size = 1

path_validate = "train_split/Gun_like_objects/agunkeychainR0P0Y0_0"
test_dataset = data_generate.VideoFrameDataset(path_validate, 196)
test_loader = DataLoader(test_dataset)

for idx,data in enumerate(test_loader):
   print(idx,data)
   print(type(data))