from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import os
from ShipDataset import ShipDataset

from transforms import DualCompose, HorizontalFlip, VerticalFlip, RandomCrop, CenterCrop

ROOT_DIR = '/kaggle/input/airbus-ship-detection'
TRAIN_DIR = os.path.join(ROOT_DIR, 'train_v2')
TEST_DIR = os.path.join(ROOT_DIR, 'test_v2')
SEGMENTATION_FILENAME = os.path.join(ROOT_DIR, 'train_ship_segmentations_v2.csv')

seg_df = pd.read_csv(SEGMENTATION_FILENAME)

unique_img_ids = seg_df[~seg_df['EncodedPixels'].isna()].groupby(['ImageId']).size().reset_index(name='counts')

np.random.seed(42)

num_ship_imgs = len(seg_df[~seg_df['EncodedPixels'].isna()])
val_ids = np.random.randint(0, num_ship_imgs, (int(num_ship_imgs * 0.1), ))
train_ids = np.setdiff1d(np.arange(num_ship_imgs), val_ids)

train_df = seg_df[~seg_df['EncodedPixels'].isna()].iloc[train_ids]
val_df = seg_df[~seg_df['EncodedPixels'].isna()].iloc[val_ids]

train_df = pd.merge(train_df, unique_img_ids)
val_df = pd.merge(val_df, unique_img_ids)





TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 2

transforms = {
    'train': DualCompose([
        HorizontalFlip(.25),
        VerticalFlip(.25),
        RandomCrop((256,256,3))
    ]),
    'val': DualCompose([
        CenterCrop((512,512,3))
    ])
}




# creating dataset and dataloaders for train, validation and test images
train_dataset = ShipDataset(TRAIN_DIR, train_df, 'train', transforms['train'])
val_dataset = ShipDataset(TRAIN_DIR, val_df, 'val', transforms['val'])

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=VAL_BATCH_SIZE, num_workers=0)

test_list_imgs = os.listdir(TEST_DIR)

test_df = pd.DataFrame({
    'ImageId': test_list_imgs,
    'EncodedPixels': None
})


test_dataset = ShipDataset(TEST_DIR, test_df, 'test', None)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=TEST_BATCH_SIZE, num_workers=0)