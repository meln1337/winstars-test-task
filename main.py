import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import segmentation_models_pytorch as smp

from losses import get_loss
from loaders import train_loader, val_loader, test_loader, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE

# Initialize parser
parser = argparse.ArgumentParser(
    prog='Winstars Test Task',
    description='The program segments ships',
    epilog='Text at the bottom of help'
)

# Adding optional argument
parser.add_argument("-lr", "--learning_rate",
                    type=float, default=1e-4, help="Learning rate")
parser.add_argument("-n_epochs", "--number_of_epochs",
                    type=int, default=1, help="Number of epochs to train")
parser.add_argument("-device",
                    type=str, default="cpu", help="The device to train on", choices=['cuda', 'cpu'])
parser.add_argument("-root_dir", "--root_directory",
                    type=str, default="/", help="Root directory")
parser.add_argument("-train_dir", "--train_directory",
                    type=str, default="train_v2", help="Train directory")
parser.add_argument("-test_dir", "--test_directory",
                    type=str, default="train_v2", help="Test directory")
parser.add_argument("-encoder", "--encoder_name",
                    type=str, default="resnet34", help="The model to use as encoder",
                    choices=['resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument("-weights", "--encoder_weights",
                    type=str, default="imagenet", help="Encoder weights")
parser.add_argument("-criterion",
                    type=str, default="BCEWithDigits", help="Criterion",
                    choices=['BCEWithDigits', 'FocalLossWithDigits', 'BCEDiceWithLogitsLoss', 'BCEJaccardWithLogitsLoss'])

# Read arguments from command line
args = parser.parse_args()

if (args.device == 'cuda' and torch.cuda.is_available()):
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Loading pretrained Unet
model = smp.Unet(args.encoder, encoder_weights="imagenet", activation=None).to(DEVICE)


criterion = get_loss(args.criterion)

run_id = 1

train(
    model=model,
    criterion=criterion,
    optimizer=optim.Adam(model.parameters(), lr=args.learning_rate),
    train_loader=train_loader,
    valid_loader=val_loader,
    train_batch_size= TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    fold=run_id,
    n_epochs = args.num_of_epochs,
    device=DEVICE
)