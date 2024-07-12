import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
from metrics import metrics

from utils import rle_decode

import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ShipDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame, mode: str = 'train', transforms=None):
        self.root_dir = root_dir
        self.mode = mode
        self.df = df

        self.img_list = self.df['ImageId'].unique()

        self.transforms = transforms
        self.posttransform = T.Compose([
            # T.ToImage(),
            # T.ToDtype(torch.float32, scale=True),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # use mean and std from ImageNet
        ])  # here we just converting to tensor and normalizing the image

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'test':
            return self.posttransform(img), str(img_path)

        encoded_mask = self.df[self.df['ImageId'] == self.img_list[idx]]['EncodedPixels'].values[0]
        mask = rle_decode(encoded_mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return self.posttransform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()

    def __len__(self) -> int:
        return len(self.img_list)