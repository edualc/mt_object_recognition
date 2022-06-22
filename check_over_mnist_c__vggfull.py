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
from lateral_connections.model_factory import *

import datetime

mnist_c_variants = [
    'pixelate', 'dotted_line', 'gaussian_blur', 'elastic_transform', 'jpeg_compression', 'speckle_noise', 'identity',
    'glass_blur', 'spatter', 'translate', 'fog', 'shear', 'scale', 'zigzag', 'defocus_blur', 'gaussian_noise',
    'contrast', 'canny_edges', 'zoom_blur', 'line', 'pessimal_noise', 'rotate', 'brightness', 'shot_noise',
    'saturate', 'motion_blur', 'snow', 'inverse', 'impulse_noise', 'stripe', 'quantize', 'frost'
]

def check_mnist_c(identifier):
    models = [
        ('vgg_full', 'models/vgg_reconstructed_lcl/2022-06-14_210747_VGG19_Full__it3750_e0.pt'),
        ('vgg_full', 'models/vgg_reconstructed_lcl/2022-06-14_210704_VGG19_Full__it21250_e4.pt'),
        ('vgg_full', 'models/vgg_reconstructed_lcl/2022-06-14_210621_VGG19_Full__it17500_e3.pt'),
        ('vgg_full', 'models/vgg_reconstructed_lcl/2022-06-14_190734_VGG19_Full__it27500_e5.pt'),
        ('vgg_full', 'models/vgg_reconstructed_lcl/2022-06-14_190720_VGG19_Full__it20000_e4.pt'),
        ('vgg_full', 'models/vgg_reconstructed_lcl/2022-06-14_190639_VGG19_Full__it11250_e2.pt'),
    ]

    data = []

    for model_key, model_path in tqdm(models, desc='Models'):
        config = get_config_by_key(model_key)
        model = load_model_by_key(model_key, model_path=model_path)

        for variant in tqdm(mnist_c_variants, desc='MNIST-C Variants', leave=False):
            dataset = load_mnistc(variant)
            loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)
            c_acc, c_loss = model.test(loader)

            model_file = model_path.split('/')[-1]
            print(f"{model_file.ljust(45)}\t{variant.ljust(20)}\t{round(c_acc,4)}")
            data.append({
                'model': model_file,
                'model_key': model_key,
                'mnist_c_variant': variant,
                'accuracy': c_acc
            })

            df = pd.DataFrame(data)
            df.to_csv('mnist_c__' + identifier + '.csv')

    df = pd.DataFrame(data)
    df.to_csv('mnist_c__' + identifier + '.csv')

def main():
    check_mnist_c('vgg_full')

if __name__ == '__main__':
    main()