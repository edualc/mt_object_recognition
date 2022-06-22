import argparse

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections import LateralModel, VggModel
from lateral_connections import VggWithLCL
from lateral_connections import MNISTCDataset
from lateral_connections.loaders import get_loaders
from lateral_connections.character_models import VGGReconstructionLCL, VggFull

import wandb
import datetime

DO_WANDB = True

def main(args):
    config = {
        'num_classes': 10,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'mnistc_corruption': 'gaussian_noise',
        'model_path': args.model_path,
    }

    train_network(config)

def train_network(config):
    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    wandb_run_name = base_name + '_VGG19_Full'
    wandb_group_name = 'VGG19_Full'

    if DO_WANDB:
        wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group=wandb_group_name,
            # group='debug',
            name=wandb_run_name,
            config=config,
            # mode='disabled',
        )

    vgg = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=0.2)
    vgg.load(config['model_path'])

    model = VggFull(vgg, learning_rate=config['learning_rate'], run_identifier=wandb_run_name)
    del vgg
    print(model)
    wandb.watch(model)

    train_loader, val_loader, test_loader, corrupt_loader = get_loaders(config['batch_size'], corruption=config['mnistc_corruption'])

    model.train_with_loader(train_loader, val_loader, test_loader=corrupt_loader, num_epochs=config['num_epochs'])

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[VGG19_Reconstructed] MNIST-C '{config['mnistc_corruption']}':\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

    # import code; code.interact(local=dict(globals(), **locals()))
    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate of VGG19\'s optimizer')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs trained')
    # WandB: https://wandb.ai/lehl/MT_LateralConnections/runs/1d4yqrn9?workspace=user-lehl
    parser.add_argument('--model_path', type=str, default='models/vgg_with_lcl/VGG19_2022-04-04_183636__it13750_e2.pt', help='Vgg19 pretrained model to use')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of models to train')

    args = parser.parse_args()

    # import code; code.interact(local=dict(globals(), **locals()))
    for _ in range(args.num_runs):
        main(args)


