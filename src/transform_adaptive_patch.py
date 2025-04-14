# Created by MacBook Pro at 13.04.25
import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor

class AdaptivePatcher:
    def __init__(self, threshold=0.08, min_size=16, output_size=(16, 16)):
        self.threshold = threshold
        self.min_size = min_size
        self.output_size = output_size
        self.to_tensor = ToTensor()

    def edge_density(self, patch):
        edges = cv2.Canny(patch, 100, 200)
        return np.mean(edges > 0)

    def quadtree_split(self, img):
        h, w = img.shape[:2]
        patches = []

        def split(x, y, width, height):
            patch = img[y:y+height, x:x+width]
            if width <= self.min_size or height <= self.min_size:
                patches.append((x, y, width, height))
                return
            if self.edge_density(patch) < self.threshold:
                patches.append((x, y, width, height))
            else:
                half_w, half_h = width // 2, height // 2
                split(x, y, half_w, half_h)
                split(x + half_w, y, half_w, half_h)
                split(x, y + half_h, half_w, half_h)
                split(x + half_w, y + half_h, half_w, half_h)

        split(0, 0, w, h)
        return patches

    def __call__(self, pil_image):
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        patch_coords = self.quadtree_split(gray)

        patch_tensors = []
        for x, y, w, h in patch_coords:
            patch = img[y:y+h, x:x+w]
            patch_resized = cv2.resize(patch, self.output_size)
            patch_tensor = self.to_tensor(patch_resized)
            patch_tensors.append(patch_tensor)

        return torch.stack(patch_tensors), patch_coords