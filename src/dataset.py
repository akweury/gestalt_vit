# Created by MacBook Pro at 13.04.25

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config

class GestaltDataset(Dataset):
    def __init__(self, root_dir, principle, split='train', transform=None):
        """
        root_dir: path to 'data/raw_patterns/res_224'
        principle: e.g., 'closure', 'proximity'
        split: 'train' or 'test'
        transform: optional image transform (e.g., ToTensor, Resize)
        """
        self.base_path = config.root / os.path.join(root_dir, principle, split)
        self.transform = transform
        self.samples = []

        # Walk the folder tree and collect samples
        for pattern_dir in glob.glob(os.path.join(str(self.base_path), "*")):
            for class_name in ['positive', 'negative']:
                class_dir = os.path.join(pattern_dir, class_name)
                label = 1 if class_name == 'positive' else 0
                image_paths = glob.glob(os.path.join(class_dir, '*.png'))
                for img_path in image_paths:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    # In dataset.py
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            patch_tensor, patch_coords = self.transform(image)
        else:
            patch_tensor, patch_coords = image, None

        return patch_tensor, label, img_path, patch_coords

