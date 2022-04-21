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
from lateral_connections.character_models import SmallVggWithLCL

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
    'identity',
    # 'dotted_line',
    # 'impulse_noise',
    # 'line',
    # 'gaussian_noise',
]

checkpoints = {
    'vggonly': [
        # These are the models with lowest val_loss:
        # 'VGG19_2022-04-04_175945__it15000_e3.pt',
        # 'VGG19_2022-04-04_183636__it13750_e2.pt',
        # 'VGG19_2022-04-04_191333__it18750_e3.pt',
        # 'VGG19_2022-04-04_160910__it17500_e3.pt',
        # 'VGG19_2022-04-04_172253__it16250_e3.pt',
        # 'VGG19_2022-04-04_164603__it18750_e3.pt',

        # These are the models with early stopping @3:
        'VGG19_2022-04-04_160910__it8750_e1.pt', 'VGG19_2022-04-04_164603__it18750_e3.pt',
        'VGG19_2022-04-04_172253__it11250_e2.pt', 'VGG19_2022-04-04_175945__it12500_e2.pt',
        'VGG19_2022-04-04_183636__it16250_e3.pt', 'VGG19_2022-04-04_191333__it10000_e2.pt'
    ],
    'lcl': [
        'VGG19_LCL_2022-04-05_155542__it12500_e2.pt', 'VGG19_LCL_2022-04-05_155601__it17500_e3.pt',
        'VGG19_LCL_2022-04-05_205611__it18750_e3.pt', 'VGG19_LCL_2022-04-05_214333__it18750_e3.pt',
        'VGG19_LCL_2022-04-06_015226__it13750_e2.pt', 'VGG19_LCL_2022-04-06_023811__it16250_e3.pt'
    ],
    'vgg16_lcl': [
        'VGG16_LCL_2022-04-07_112652__it18750_e3.pt','VGG16_LCL_2022-04-07_112710__it12500_e2.pt',
        'VGG16_LCL_2022-04-07_202405__it15000_e3.pt','VGG16_LCL_2022-04-07_202410__it17500_e3.pt',
        'VGG16_LCL_2022-04-08_013150__it18750_e3.pt','VGG16_LCL_2022-04-08_020051__it11250_e2.pt'
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
    },
    'vgg16_lcl': {
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

def check_mnist_c(identifier, run_idx=None):
    config = configs[identifier]
    model_files = checkpoints[identifier]

    if run_idx:
        filtered_model_files = []
        for model_index in [int(x) for x in run_idx]:
            filtered_model_files.append(model_files[model_index])
        model_files = filtered_model_files
    
    data = []

    for model_file in tqdm(model_files, desc='Models'):
        model_path = 'models/vgg_with_lcl/' + model_file

        if identifier == 'vgg16_lcl':
            model = SmallVggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
                num_multiplex=config['num_multiplex'], do_wandb=False, run_identifier="",
                lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])

        else:
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
            if run_idx:
                df.to_csv('mnist_c__' + identifier + ''.join(run_idx) + '.csv')
            else:
                df.to_csv('mnist_c__' + identifier + '.csv')

    df = pd.DataFrame(data)
    if run_idx:
        df.to_csv('mnist_c__' + identifier + ''.join(run_idx) + '.csv')
    else:
        df.to_csv('mnist_c__' + identifier + '.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lcl', default=False, action='store_true')
    parser.add_argument('--vgg16_lcl', default=False, action='store_true')
    parser.add_argument('--run_ids', nargs='+', default=None)
    args = parser.parse_args()    

    if args.lcl:
        check_mnist_c('lcl', args.run_ids)
    elif args.vgg16_lcl:
        check_mnist_c('vgg16_lcl', args.run_ids)
    else:
        check_mnist_c('vggonly', args.run_ids)


if __name__ == '__main__':
    main()