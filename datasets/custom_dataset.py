import os
from PIL import Image
import pandas as pd
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class CarsDataset(Dataset):
    def __init__(self, dataframe, transform=None, split='train', classify_by_year=True):
        self.dataframe = dataframe
        self.transform = transform
        self.split = split
        self.classify_by_year = classify_by_year

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0] # Assuming file paths are in the first column
        image = Image.open(img_name).convert('RGB')
        # Open image and convert to RGB mode
        
        if self.transform:
            image = self.transform(image)
        image_tensor = ToTensor()(image)
        label = self.dataframe.iloc[idx, 3]  # Assuming labels are in the second column
        
        if self.classify_by_year:
            # Convert label (year) to tensor
            label = torch.tensor(label, dtype=torch.float32)  # Assuming label is a numeric value
        
        return image_tensor, label