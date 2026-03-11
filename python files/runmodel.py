import os
import cv2
import glob
from PIL import Image
import numpy as np
import pickle
import shutil
from collections import Counter

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SequentialSampler, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from opacus.layers import DPLSTM
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import warnings
warnings.filterwarnings('ignore')

from util import testdata_generate, train, cnnlstm, cnn3d, DPtrain, slicer

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

item = "Glock project V1 (Updated Draft).stl"
path_validate = "temp"

os.mkdir(path_validate)
slicer.slice(item, path_validate)

test_dataset = testdata_generate.VideoFrameDataset(path_validate, 1)
test_loader = DataLoader(test_dataset)

with open('models/save/cnnlstm_e5_b8ed7038fcc611f09084bab6975ba98b.pkl', 'rb') as f:
    model = pickle.load(f)
model.to(device)
model_name = "CNNLSTM"

for inputs, targets, masks, max_seq, position_intervals in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            break

def predict(model, loader):
    model.eval()
    predictions = []
    confidences = [] # List to store confidence scores
    
    with torch.no_grad():
        for batch in loader: 
            inputs, targets, masks, max_seq, position_intervals = batch
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            outputs = model(inputs, masks)
            
            # Apply Softmax to get probabilities (0.0 to 1.0)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the highest probability and its index
            conf, predicted = torch.max(probabilities, 1)
            
            predictions.append(predicted)
            confidences.append(conf)
            
    return predictions, confidences

# Run the prediction
output, confidence_tensors = predict(model, test_loader)

shutil.rmtree(path_validate)

# Convert GPU tensors to standard Python values
predicted_indices = [p.item() for p in output]
# Extract confidence as a percentage
confidence_scores = [c.item() * 100 for c in confidence_tensors]

print(f"Segment Predictions: {predicted_indices}")
print(f"Segment Confidences: {[f'{c:.2f}%' for c in confidence_scores]}")

# --- Final Prediction Logic ---
counts = Counter(predicted_indices)
final_class_index = counts.most_common(1)[0][0]

# Calculate Average Confidence for the chosen class
# (Filter scores that belong only to the most predicted class)
counts = Counter(predicted_indices)
final_class_index = counts.most_common(1)[0][0]

final_conf = np.mean([score for idx, score in zip(predicted_indices, confidence_scores) if idx == final_class_index])
class_names = ['Cup', 'FathersDay', 'Gun_like_objects', 'GunPart', 'Horse', 'Pistol', 'PortalGun', 'Revolver', 'SpecialRevolver', 'ToyGun'] 
print("-" * 30)
print(f"Final Predicted Class Index: {final_class_index}")
print(f"Predicted Label: {class_names[final_class_index]}")
print(f"Confidence Score: {final_conf:.2f}%")