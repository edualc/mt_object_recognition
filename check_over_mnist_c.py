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

import datetime

mnist_c_variants = [ 'line','elastic_transform','zigzag','motion_blur',
    'gaussian_blur','pessimal_noise','impulse_noise','defocus_blur',
    'spatter','dotted_line','gaussian_noise','pixelate','zoom_blur',
    'shot_noise','fog' ]

checkpoints = {
    'vggonly': [
        'VGG19_2022-04-04_160910__it8750_e1.pt', 'VGG19_2022-04-04_164603__it18750_e3.pt',
        'VGG19_2022-04-04_172253__it11250_e2.pt', 'VGG19_2022-04-04_175945__it12500_e2.pt',
        'VGG19_2022-04-04_183636__it16250_e3.pt', 'VGG19_2022-04-04_191333__it10000_e2.pt'
    ],
    'lcl': [
        'VGG19_LCL_2022-04-05_155542__it12500_e2.pt', 'VGG19_LCL_2022-04-05_155601__it17500_e3.pt',
        'VGG19_LCL_2022-04-05_205611__it18750_e3.pt', 'VGG19_LCL_2022-04-05_214333__it18750_e3.pt',
        'VGG19_LCL_2022-04-06_015226__it13750_e2.pt', 'VGG19_LCL_2022-04-06_023811__it16250_e3.pt'
    ]
}

configs = {
    'vggonly': {
        'num_classes': 10,
        'learning_rate': 3e-4,
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
    'lcl': {
        'num_classes': 10,
        'learning_rate': 1e-3,
        'dropout': 0.2,
        'num_epochs': 4,
        'batch_size': 10,
        'use_lcl': True,
        'num_multiplex': 4,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.0,
        'lcl_iota': 0.2
    }
}

def check_mnist_c(identifier):
    config = configs[identifier]
    model_files = checkpoints[identifier]

    data = []

    for model_file in tqdm(model_files, desc='Models'):
        model_path = 'models/vgg_with_lcl/' + model_file

        model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
            num_multiplex=config['num_multiplex'], do_wandb=False, run_identifier="",
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        model.load(model_path)

        if config['use_lcl']:
            model.features.lcl3.enable()

        for variant in tqdm(mnist_c_variants, desc='MNIST-C Variants', leave=False):
            dataset = load_mnistc(variant)
            loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)
            c_acc, c_loss = model.test(loader)

            print(f"{model_file.ljust(45)}\t{variant.ljust(20)}\t{round(c_acc,4)}")
            data.append({
                'model': model_file,
                'mnist_c_variant': variant,
                'accuracy': c_acc
            })

    df = pd.DataFrame(data)
    df.to_csv('mnist_c__' + identifier + '.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lcl', default=False, action='store_true')
    args = parser.parse_args()    

    if args.lcl:
        check_mnist_c('lcl')
    else:
        check_mnist_c('vggonly')


if __name__ == '__main__':
    main()