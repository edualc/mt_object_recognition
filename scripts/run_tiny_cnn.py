import argparse

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections.character_models import TinyCNN 
from lateral_connections import MNISTCDataset
from lateral_connections.loaders import get_loaders
from lateral_connections.model_factory import load_model_by_key

import wandb
import datetime

DO_WANDB = True

def main(args):
    config = {
        'batch_size': args.batch_size,
        'conv_channels': 10,
        'learning_rate': args.lr,
        'num_classes': 10,
        'num_epochs': args.num_epochs,
    }

    train_network(config)

def train_network(config):
    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wandb_run_name = 'TinyCNN_' + base_name
    config['run_identifier'] = wandb_run_name

    if DO_WANDB:
        wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group='TinyCNN',
            # group='debug',
            name=wandb_run_name,
            config=config,
            # mode='disabled',
        )

    model = load_model_by_key('tiny_cnn', config=config)
    wandb.watch(model)

    train_loader, val_loader, test_loader, corrupt_loader = get_loaders(config['batch_size'])
    
    model.train_with_loader(train_loader, val_loader, test_loader=corrupt_loader, num_epochs=config['num_epochs'])

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[{'TinyCNN'}] MNIST-C:\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

    # import code; code.interact(local=dict(globals(), **locals()))
    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs trained')

    args = parser.parse_args()
    main(args)
    # import code; code.interact(local=dict(globals(), **locals()))


