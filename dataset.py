import os
import cv2
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MyDataset(Dataset):
    def __init__(self, root_dir, color_dict, transform=None):
        self.transform = transform
        self.color_dict = color_dict
        self.images = []
        self.masks = []
        list_file = os.listdir(os.path.join(root_dir, "images"))
        for file_name in list_file:
            name, ext = os.path.split(file_name)
            if ext in ['.jpg', '.jpeg', '.png']:
                self.images.append(os.path.join(root_dir, "images", file_name))
                self.masks.append(os.path.join(root_dir, "labels", name+'.png'))

    def __getitem__(self, index):
        try:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            index=index-1
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.color_map(mask, self.color_dict)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def color_map(mask, color_dict):
        color_mask = np.zeros([mask.shape[0], mask.shape[1]])
        for idx, item in enumerate(color_dict):
            check = mask == color_dict[item]
            check = np.logical_and(np.logical_and(check[:,:,0], check[:,:,1]), check[:,:,2])
            color_mask[check]= idx+1
        
        return np.uint8(color_mask)
