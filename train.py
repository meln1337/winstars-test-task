import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from pathlib import Path
import random
import datetime
from metrics import metrics

import json

from utils import write_event

def train(
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: nn.Module,
        train_batch_size: int = 16,
        val_batch_size: int = 4,
        n_epochs: int = 1,
        fold: int = 1,
        device: torch.device = 'cpu'
):
    model_path = Path(f'model_{fold}.pt')

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print(f'Restored model, epoch {epoch}, step {step}')
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 50
    log = open('train_{fold}.log'.format(fold=fold), 'at', encoding='utf8')

    model = model.to(device)

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        train_loop = tqdm(total=len(train_loader) * train_batch_size)
        train_loop.set_description(f'Epoch {epoch}')
        losses = []
        valid_metrics = metrics(batch_size=val_batch_size)  # for validation

        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                train_loop.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                train_loop.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            train_loop.close()
            save(epoch + 1)

            # Validation
            comb_loss_metrics = validation(model, criterion, valid_loader, valid_metrics, device)
            write_event(log, step, **comb_loss_metrics)

        except KeyboardInterrupt:
            train_loop.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def validation(
        model: nn.Module,
        criterion: nn.Module,
        valid_loader: DataLoader,
        metrics,
        device: torch.device
):
    print("Validation")

    losses = []
    model.eval()

    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        metrics.collect(outputs.detach().cpu(), targets.detach().cpu())  # get metrics

    valid_loss = np.mean(losses)  # float
    valid_dice, valid_jaccard = metrics.get()  # float

    print('Valid loss: {:.5f}, Jaccard: {:.5f}, Dice: {:.5f}'.format(valid_loss, valid_jaccard, valid_dice))
    comb_loss_metrics = {'valid_loss': valid_loss, 'jaccard': valid_jaccard.item(), 'dice': valid_dice.item()}

    return comb_loss_metrics
