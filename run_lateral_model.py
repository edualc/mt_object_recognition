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

DO_WANDB = True

def main(args):
    config = {
        'num_classes': 10,
        'learning_rate': args.lr,
        'dropout': args.dropout,
        'num_multiplex': args.num_multiplex,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'use_lcl': args.lcl,
        'lcl_alpha': args.lcl_alpha,
        'lcl_eta': args.lcl_eta,
        'lcl_theta': args.lcl_theta,
        'lcl_iota': args.lcl_iota,
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
    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    if config['use_lcl']:
        wandb_run_name = 'VGG19_LCL_' + base_name
    else:
        wandb_run_name = 'VGG19_' + base_name

    if DO_WANDB:
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group='debug',
            # group='Vgg19WithLCL' if config['use_lcl'] else 'Vgg19',
            name=wandb_run_name,
            config=config
        )

    model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
        num_multiplex=config['num_multiplex'], do_wandb=DO_WANDB, run_identifier=wandb_run_name,
        lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
    
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
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs trained')
    parser.add_argument('--lcl', default=False, action='store_true', help='Whether VGG19 should be trained with or without LCL')
    parser.add_argument('--lcl_alpha', type=float, default=0.1, help='Rate at which kernel K is changed by K_change')
    parser.add_argument('--lcl_eta', type=float, default=0.1, help='Rate at which the output is changed by O=(1-eta)*A+eta*L')
    parser.add_argument('--lcl_theta', type=float, default=0.1, help='How much the noise is added to the LCL training (breaking symmetry)')
    parser.add_argument('--lcl_iota', type=float, default=0.1, help='Rate at which the argmax((1-iota)*A+iota*L) is calculated to determine active multiplex cells')

    args = parser.parse_args()
    main(args)
    # import code; code.interact(local=dict(globals(), **locals()))


