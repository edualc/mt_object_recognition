import argparse

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections import LateralModel, VggModel
from lateral_connections import SmallVggWithLCL
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
        'lcl_alpha': args.lcl_alpha,
        'lcl_eta': args.lcl_eta,
        'lcl_theta': args.lcl_theta,
        'lcl_iota': args.lcl_iota,
    }

    train_network(config)

def eval_network(config, model_path):
    model = SmallVggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'], num_multiplex=config['num_multiplex'], do_wandb=False)
    model.load(model_path)

    _, _, test_loader, corrupt_loader = get_loaders(config['batch_size'])
    n_acc, n_loss = model.test(test_loader)
    print(f"[VGG16+LCL'] MNIST:\t\tAccuracy:{n_acc:1.4f}\tLoss:{n_loss:1.4f}")

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[VGG16+LCL] MNIST-C:\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

def train_network(config):
    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wandb_run_name = 'VGG16_LCL_' + base_name

    if DO_WANDB:
        wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group='Vgg16WithLCL_eval',
            #group='debug',
            name=wandb_run_name,
            config=config
        )

    model = SmallVggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
        num_multiplex=config['num_multiplex'], do_wandb=DO_WANDB, run_identifier=wandb_run_name,
        lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
    
    train_loader, val_loader, test_loader, corrupt_loader = get_loaders(config['batch_size'])
    
    model.train_with_loader(train_loader, val_loader, test_loader=corrupt_loader, num_epochs=config['num_epochs'])

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[{'VGG16+LCL'}] MNIST-C:\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout percentage of VGG16')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of VGG16\'s optimizer')
    parser.add_argument('--num_multiplex', type=int, default=4, help='Number of multiplex cells in LCL layers')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs trained')
    parser.add_argument('--lcl_alpha', type=float, default=1e-3, help='Rate at which kernel K is changed by K_change')
    parser.add_argument('--lcl_eta', type=float, default=0.0, help='Rate at which the output is changed by O=(1-eta)*A+eta*L')
    parser.add_argument('--lcl_theta', type=float, default=0.2, help='How much the noise is added to the LCL training (breaking symmetry)')
    parser.add_argument('--lcl_iota', type=float, default=0.2, help='Rate at which the argmax((1-iota)*A+iota*L) is calculated to determine active multiplex cells')

    args = parser.parse_args()

    for _ in range(3):
        main(args)
    # import code; code.interact(local=dict(globals(), **locals()))


