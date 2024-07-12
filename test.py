import torch
import torch.nn.functional as F
# import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import cv2
import argparse
import torchvision.transforms as T
from ShipDataset import IMAGENET_STD, IMAGENET_MEAN
from skimage.morphology import binary_opening, disk

import os

# Initialize parser
parser = argparse.ArgumentParser(
    prog='Application of the model',
    description='The program segments ships',
    epilog='Specify needed parameters'
)


# Adding optional argument
parser.add_argument("-img_path",
                    type=str, default='./kaggle/input/airbus-ship-detection/test_v2/0010551d9.jpg', help="Path to the model")

# Read arguments from command line
args = parser.parse_args()
img_path = args.img_path

if not os.path.exists(img_path):
    raise FileNotFoundError('This file does not exists')

img_init = cv2.imread(img_path)
img_init = cv2.cvtColor(img_init, cv2.COLOR_BGR2RGB)

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # use mean and std from ImageNet
])

if img_init.shape[0] != 768 or img_init.shape[1] != 768:
    img_init = cv2.resize(img_init, (768, 768))

img = transforms(img_init).unsqueeze(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'model_scripted_cuda.pt'

model = torch.jit.load(model_path)
model.eval()

with torch.no_grad():
    img = img.to(DEVICE)
    output = model(img)

    mask = F.sigmoid(output[0, 0]).data.detach().cpu().numpy()
    mask = binary_opening(mask > 0.5, disk(2))

plt.imshow(img_init)
plt.imshow(mask, alpha=.25)
plt.axis('off')
plt.title(f'Predicted mask for {img_path.split("/")[-1]} ({"no ships" if mask.sum() == 0 else "has ships"})')
plt.show()