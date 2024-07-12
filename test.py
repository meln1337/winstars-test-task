import torch
import segmentation_models_pytorch as smp

DEVICE = torch.device('cuda' if torch.cuda.is_available())

model_path = 'model.pt'

model = smp.Unet('resnet34', encoder_weights="imagenet", activation=None).to(DEVICE)

state = torch.load(model_path).to(DEVICE)
epoch = state['epoch']
step = state['step']
model.load_state_dict(state['model'])
print(f'Restored model, epoch {epoch}, step {step}')