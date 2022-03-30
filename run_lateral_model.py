import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections import LateralModel, VggModel
from lateral_connections import VggWithLCL
from lateral_connections import MNISTCDataset

def main():
    num_classes = 10

    model = VggWithLCL(num_classes, learning_rate=0.003, dropout=0.2, num_multiplex=4)
    model.features.lcl3.enable()

    train_loader = torch.utils.data.DataLoader(load_mnist(train=True), batch_size=20, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(load_mnist(), batch_size=20, shuffle=False, num_workers=1)
    corrupt_loader = torch.utils.data.DataLoader(load_mnistc(), batch_size=20, shuffle=False, num_workers=1)
    
    model.train_with_loader(train_loader, num_epochs=1)
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
    dataset = MNISTCDataset(images, labels, transform=image_transform())

    return dataset

def load_mnist(train=False):
    dataset = MNIST('images/mnist/', train=train, transform=image_transform(), download=True)

    return dataset

if __name__ == '__main__':
    main()
