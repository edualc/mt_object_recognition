import argparse

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections import LateralModel, VggModel
from lateral_connections import VggWithLCL
from lateral_connections import MNISTCDataset
from lateral_connections.loaders import get_loaders, load_mnistc
from lateral_connections.character_models import SmallVggWithLCL, VGGReconstructionLCL

import datetime

mnist_c_variants = [ 
    # 'zigzag',
    # 'elastic_transform',
    # 'motion_blur',
    # 'gaussian_blur',
    # 'defocus_blur',
    # 'pixelate',
    # 'zoom_blur',
    # 'fog',
    
    # 'spatter',
    # 'shot_noise',
    # 'pessimal_noise',
    # 'identity',
    # 'dotted_line',
    # 'impulse_noise',
    # 'line',
    # 'gaussian_noise',

    'pixelate', 'dotted_line', 'gaussian_blur', 'elastic_transform', 'jpeg_compression', 'speckle_noise', 'identity',
    'glass_blur', 'spatter', 'translate', 'fog', 'shear', 'scale', 'zigzag', 'defocus_blur', 'gaussian_noise',
    'contrast', 'canny_edges', 'zoom_blur', 'line', 'pessimal_noise', 'rotate', 'brightness', 'shot_noise',
    'saturate', 'motion_blur', 'snow', 'inverse', 'impulse_noise', 'stripe', 'quantize', 'frost'
]

checkpoints = {
    'vggonly': 'models/vgg_with_lcl/VGG19_2022-04-04_183636__it16250_e3.pt',
    'vgg19r_lcl5': 'models/vgg_reconstructed_lcl/2022-04-24_004459_LCL5_d2__it23750_e4.pt'
}

configs = {
    'vggonly': {
        'num_classes': 10,
        'learning_rate': 1e-3,
        'dropout': 0.2,
        'num_epochs': 4,
        'batch_size': 10,
        'use_lcl': False,
        'num_multiplex': 4,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.0,
        'lcl_iota': 0.2
    },
    'vgg19r_lcl5': {
        'num_classes': 10,
        'learning_rate': 1e-4,
        'num_multiplex': 4,
        'batch_size': 10,
        'num_epochs': 5,
        'lcl_alpha': 3e-4,
        'lcl_eta': 0.01,
        'lcl_theta': 0.2,
        'lcl_iota': 0.2,
        'lcl_distance': 2,
        'lcl_k': 5,
        'after_pooling': 5
    }
}

def check_mnist_c(identifier, variant):
    config = configs[identifier]
    model_path = checkpoints[identifier]

    data = []

    if identifier == 'vgg19r_lcl5':
        vgg = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=0.2)
        model = VGGReconstructionLCL(vgg, learning_rate=config['learning_rate'], after_pooling=config['after_pooling'],
            num_multiplex=config['num_multiplex'], run_identifier='', lcl_distance=config['lcl_distance'],
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        model.load(model_path)
        model.features.lcl.enable()
        del vgg

    else:
        model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
            num_multiplex=config['num_multiplex'], do_wandb=False, run_identifier="",
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        model.load(model_path)


    dataset = load_mnistc(variant)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)

    for i, (images, labels) in tqdm(enumerate(loader, 0), total=len(loader), desc='Testing'):

        start_index = i * config['batch_size']

        images = images.to(model.device)
        labels = labels.to(model.device)

        _, predictions = torch.max(model(images), 1)

        for k in range(config['batch_size']):
            data.append({
                'dataset': variant,
                'experiment': identifier,
                'model_path': model_path,
                'image_index': start_index + k,
                'label': labels[k].item(),
                'prediction': predictions[k].item()
            })

        df = pd.DataFrame(data)
        df.to_csv('mnist_c__' + variant + '__' + identifier + '__predictions.csv', index=False)

    df = pd.DataFrame(data)
    df.to_csv('mnist_c__' + variant + '__' + identifier + '__predictions.csv', index=False)

def main():
    # check_mnist_c('vgg19r_lcl5', 'gaussian_noise')
    check_mnist_c('vggonly', 'gaussian_noise')
    # check_mnist_c('vgg19r_lcl5', 'identity')
    # check_mnist_c('vgg19r_lcl5', 'line')


if __name__ == '__main__':
    main()