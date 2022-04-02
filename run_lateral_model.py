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

import wandb
import datetime

DO_WANDB = False

def main(args):
    config = {
        'num_classes': 10,
        'learning_rate': args.lr,
        'dropout': args.dropout,
        'num_multiplex': args.num_multiplex,
        'batch_size': args.batch_size,
        'use_lcl': args.lcl,
        'num_epochs': args.num_epochs
    }

    train_network(config)
    # eval_network(use_lcl=True, model_path='models/vgg_with_lcl/2022-04-01_085941__it1000_e0.pt') # 6k iterations w/ LCL
    # eval_network(use_lcl=False, model_path='models/vgg_with_lcl/2022-04-01_103028__e0.pt') # 6k iterations w/o LCL

def eval_network(config, model_path):
    model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'], num_multiplex=config['num_multiplex'], do_wandb=False)
    model.load(model_path)

    if config['use_lcl']:
        model.features.lcl3.enable()

    _, _, test_loader, corrupt_loader = get_loaders(config['batch_size'])
    n_acc, n_loss = model.test(test_loader)
    print(f"[{'VGG19+LCL' if config['use_lcl'] else 'VGG19'}] MNIST:\t\tAccuracy:{n_acc:1.4f}\tLoss:{n_loss:1.4f}")

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[{'VGG19+LCL' if config['use_lcl'] else 'VGG19'}] MNIST-C:\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

def train_network(config):
    wandb_run_name = 'VGG19_' + 'LCL_' if config['use_lcl'] else '' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    if DO_WANDB:
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group='Vgg19WithLCL' if config['use_lcl'] else 'Vgg19',
            name=wandb_run_name,
            config=config
        )

    model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'], num_multiplex=config['num_multiplex'], do_wandb=DO_WANDB, run_identifier=wandb_run_name)
    
    if config['use_lcl']:
        model.features.lcl3.enable()

    train_loader, val_loader, test_loader, corrupt_loader = get_loaders(config['batch_size'])
    
    model.train_with_loader(train_loader, val_loader, num_epochs=config['num_epochs'])

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[{'VGG19+LCL' if config['use_lcl'] else 'VGG19'}] MNIST-C:\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

    # import code; code.interact(local=dict(globals(), **locals()))
    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout percentage of VGG19')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate of VGG19\'s optimizer')
    parser.add_argument('--num_multiplex', type=int, default=4, help='Number of multiplex cells in LCL layers')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lcl', default=False, action='store_true', help='Whether VGG19 should be trained with or without LCL')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs trained')

    args = parser.parse_args()
    main(args)
    # import code; code.interact(local=dict(globals(), **locals()))


