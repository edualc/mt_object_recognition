import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections import LateralModel, VggModel
from lateral_connections import VggWithLCL
from lateral_connections import MNISTCDataset

import wandb
import datetime

DO_WANDB = True

def main():
    num_classes = 10

    # import code; code.interact(local=dict(globals(), **locals()))

    config = {
        'learning_rate': 0.003,
        'dropout': 0.2,
        'num_multiplex': 4,
        'batch_size': 10
    }

    wandb_run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    if DO_WANDB:
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group='VggWithLCL',
            name=wandb_run_name,
            config=config
        )

    model = VggWithLCL(num_classes, learning_rate=config['learning_rate'], dropout=config['dropout'], num_multiplex=config['num_multiplex'], do_wandb=DO_WANDB)
    model.features.lcl3.enable()

    train_loader = torch.utils.data.DataLoader(load_mnist(train=True), batch_size=config['batch_size'], shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(load_mnist(), batch_size=config['batch_size'], shuffle=False, num_workers=1)
    corrupt_loader = torch.utils.data.DataLoader(load_mnistc(), batch_size=config['batch_size'], shuffle=False, num_workers=1)
    
    model.train_with_loader(train_loader, test_loader, num_epochs=1)

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"MNIST-C:\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")
    import code; code.interact(local=dict(globals(), **locals()))


def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

def load_mnistc(dirname=None, train=False):
    if dirname is None:
        dirname = 'line'
    root = 'images/mnist_c/' + dirname
    dataset_type = 'train' if train else 'test'

    images = np.load(root + '/' + dataset_type + '_images.npy')
    images  = images.transpose(0, 3, 1, 2)[:, 0, ...]

    labels = np.load(root + '/' + dataset_type + '_labels.npy').reshape((images.shape[0],))
    return MNISTCDataset(images, labels, transform=image_transform())

def load_mnist(train=False):
    return MNIST('images/mnist/', train=train, transform=image_transform(), download=True)

if __name__ == '__main__':
    main()
