# predictor.py
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from util import testdata_generate  # Ensure this is in your PYTHONPATH
import warnings
warnings.filterwarnings("ignore")

class VideoPredictor:
    def __init__(self, model_path):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.class_names = ['Cup', 'FathersDay', 'Gun_like_objects', 'GunPart', 'Horse', 
                            'Pistol', 'PortalGun', 'Revolver', 'SpecialRevolver', 'ToyGun']
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model.to(self.device)
        self.model.eval()

    def predict_video(self, video_path):
        # Initialize dataset/loader for the specific path
        print(f"DEBUG: Python is now processing: {video_path}")

        test_dataset = testdata_generate.VideoFrameDataset(video_path, 1)
        test_loader = DataLoader(test_dataset)
        
        predicted_indices = []
        confidence_scores = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets, masks, _, _ = batch
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                
                outputs = self.model(inputs, masks)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probabilities, 1)
                
                predicted_indices.append(predicted.item())
                confidence_scores.append(conf.item() * 100)

        # Final Logic
        if not predicted_indices:
            return "Unknown", 0.0

        final_class_idx = max(set(predicted_indices), key=predicted_indices.count)
        relevant_confidences = [score for idx, score in zip(predicted_indices, confidence_scores) 
                                if idx == final_class_idx]
        avg_conf = np.mean(relevant_confidences)
        
        return self.class_names[final_class_idx], float(avg_conf)