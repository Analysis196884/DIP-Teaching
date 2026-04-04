import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        self.is_train = 'train' in list_file
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        
        # Split into RGB and Semantic images first
        # Based on current code: Semantic is on the left (:256), RGB is on the right (256:)
        img_semantic = img_color_semantic[:, :256, :]
        img_rgb = img_color_semantic[:, 256:, :]
        
        if self.is_train:
            # 1. Resize to 286x286 for Random Crop
            img_rgb = cv2.resize(img_rgb, (286, 286), interpolation=cv2.INTER_LINEAR)
            img_semantic = cv2.resize(img_semantic, (286, 286), interpolation=cv2.INTER_LINEAR)
            
            # 2. Random Crop back to 256x256
            h_offset = random.randint(0, 286 - 256)
            w_offset = random.randint(0, 286 - 256)
            img_rgb = img_rgb[h_offset:h_offset+256, w_offset:w_offset+256, :]
            img_semantic = img_semantic[h_offset:h_offset+256, w_offset:w_offset+256, :]
            
            # 3. Random Horizontal Flip (p=0.5)
            if random.random() > 0.5:
                img_rgb = cv2.flip(img_rgb, 1)
                img_semantic = cv2.flip(img_semantic, 1)
        
        # Ensure arrays are contiguous after cv2 operations
        img_rgb = np.ascontiguousarray(img_rgb)
        img_semantic = np.ascontiguousarray(img_semantic)

        # Convert the images to PyTorch tensors
        image_rgb = torch.from_numpy(img_rgb).float()
        image_semantic = torch.from_numpy(img_semantic).float()
        
        # Normalize to [-1, 1]
        image_rgb = (image_rgb / 127.5) - 1.0
        image_semantic = (image_semantic / 127.5) - 1.0
        
        # (H, W, C) -> (C, H, W)
        image_rgb = image_rgb.permute(2, 0, 1)
        image_semantic = image_semantic.permute(2, 0, 1)
        
        return image_rgb, image_semantic