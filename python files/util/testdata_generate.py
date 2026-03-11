import os
import cv2
import glob
from PIL import Image
import numpy as np


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

# insert position_interval


class VideoFrameDataset(Dataset):
    """
    A dataset that loads sequences of video frames for classification tasks.

    This dataset supports two directory structures:
    1. Standard Hierarchy: ``root_dir/class_name/video_name/frames...``
    2. Single Video: ``root_dir/frames...`` (treated as a single sequence with label 0)

    For every video, we generate ``intervals`` samples by partitioning the sequence 
    into equal segments. Within each segment we take a subset of frames and 
    pad/truncate them to a fixed length ``frames_per_video``.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root. Can be a folder of class folders or a single 
        folder containing frames.
    frames_per_video : int, optional
        Fixed number of frames returned for each sample. Default: 20.
    intervals : int, optional
        Number of segments to divide each video into. Default: 10.
    image_size : tuple of int, optional
        Target size ``(height, width)`` for resizing frames. Default: (32, 32).
    transform : callable, optional
        Optional transform to be applied on a frame.
    cache : bool, optional
        If True, decoded frames are cached in memory.

    Returns
    -------
    tuple
        (frames, label, mask, max_seq, position_interval)
    """

    def __init__(self,
                 root_dir: str,
                 frames_per_video: int = 20,
                 intervals: int = 10,
                 image_size: tuple = (32, 32),
                 transform: callable = None,
                 cache: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.intervals = intervals
        self.image_size = image_size
        self.transform = transform
        self.cache = cache
        
        self.video_files = []
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        # Check if root_dir contains frames directly (Single Video Mode)
        root_content = sorted(os.listdir(root_dir))
        root_frames = [os.path.join(root_dir, f) for f in root_content 
                       if f.lower().endswith(valid_exts)]

        if root_frames:
            # --- Single Video Mode ---
            # Treat the root directory as one video. 
            # We assign a default class name and index (0).
            self.class_map = {'default': 0}
            self.video_files.append(('default', root_frames))
            
        else:
            # --- Original Hierarchy Mode ---
            # Structure: root_dir -> class_dir -> video_dir -> frames
            
            # Build class to index mapping
            class_names = [d for d in root_content 
                           if os.path.isdir(os.path.join(root_dir, d))]
            class_names.sort()
            self.class_map = {name: i for i, name in enumerate(class_names)}

            # Build list of videos
            for class_name in class_names:
                class_dir = os.path.join(self.root_dir, class_name)
                for video_name in sorted(os.listdir(class_dir)):
                    video_path = os.path.join(class_dir, video_name)
                    if not os.path.isdir(video_path):
                        continue
                        
                    # Collect all image files
                    frames = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))
                              if f.lower().endswith(valid_exts)]
                    
                    if frames:
                        self.video_files.append((class_name, frames))

        # Optionally cache decoded frames
        self._frame_cache = {} if cache else None

    def __len__(self) -> int:
        return len(self.video_files) * self.intervals

    def _load_frame(self, path: str) -> torch.Tensor:
        """Load an image from disk, convert to grayscale and resize."""
        if self._frame_cache is not None and path in self._frame_cache:
            return self._frame_cache[path]

        with Image.open(path) as img:
            img = img.convert('L')
            if self.image_size is not None:
                img = img.resize(self.image_size, Image.BILINEAR)
            
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                arr = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0)

        if self._frame_cache is not None:
            self._frame_cache[path] = tensor
        return tensor

    def __getitem__(self, idx: int):
        video_idx = idx // self.intervals
        position_interval = idx % self.intervals
        
        class_name, frame_paths = self.video_files[video_idx]
        num_real_images = len(frame_paths)

        images_per_interval = max((num_real_images + self.intervals - 1) // self.intervals, 1)
        start_idx = position_interval * images_per_interval
        end_idx = min(start_idx + images_per_interval, num_real_images)

        frames_tensor = torch.zeros((self.frames_per_video, 1, *self.image_size), dtype=torch.float32)
        mask = torch.zeros((self.frames_per_video,), dtype=torch.float32)

        load_count = 0
        for i in range(start_idx, end_idx):
            if load_count >= self.frames_per_video:
                break
            frame_tensor = self._load_frame(frame_paths[i])
            if frame_tensor.dim() == 3:
                frames_tensor[load_count] = frame_tensor
            else:
                frames_tensor[load_count] = frame_tensor.unsqueeze(0)
            mask[load_count] = 1.0
            load_count += 1

        label = self.class_map[class_name]
        max_seq = num_real_images

        return frames_tensor, label, mask, max_seq, position_interval