import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from .dataset import MNISTCDataset

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

def get_loaders(batch_size, corruption='gaussian_noise'):
    dataset = load_mnist(train=True)
    train_size = 50000
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(load_mnist(), batch_size=batch_size, shuffle=False, num_workers=1)
    corrupt_loader = torch.utils.data.DataLoader(load_mnistc(dirname=corruption), batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, eval_loader, test_loader, corrupt_loader