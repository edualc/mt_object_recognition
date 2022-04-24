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
    'spatter',
    'shot_noise',
    'pessimal_noise',
    'identity',
    'dotted_line',
    'impulse_noise',
    'line',
    'gaussian_noise',
]

def check_mnist_c(identifier, run_idx=None):
    model_files = [

        # Reruns without symmetric padding (lr=3e-4)
        '2022-04-23_181034_LCL5_d2__it23750_e4.pt',
        '2022-04-23_181111_LCL5_d2__it12500_e2.pt',
        '2022-04-23_201731_LCL5_d2__it11250_e2.pt',
        '2022-04-24_071904_LCL5_d2__it22500_e4.pt',
        '2022-04-24_071930_LCL5_d2__it16250_e3.pt',

        #
        # '2022-04-19_194344_LCL5_d2__it21250_e4.pt',
        # '2022-04-19_194332_LCL5_d2__it16250_e3.pt',
        # '2022-04-19_194337_LCL5_d2__it21250_e4.pt',
        # '2022-04-19_194505_LCL5_d2__it11250_e2.pt',
        # '2022-04-19_081920_LCL5__it21250_e4.pt',

        # # Early Stopping 3 Checkpoints
        # '2022-04-19_194344_LCL5_d2__it21250_e4.pt', # SAME
        # '2022-04-19_194337_LCL5_d2__it12500_e2.pt',
        # '2022-04-19_194332_LCL5_d2__it5000_e1.pt', # SAME
        # '2022-04-19_194505_LCL5_d2__it11250_e2.pt', # SAME
        # '2022-04-19_081920_LCL5__it8750_e1.pt',
    ]

    config = {
        'num_classes': 10,
        'learning_rate': 3e-4,
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

    if run_idx:
        filtered_model_files = []
        for model_index in [int(x) for x in run_idx]:
            filtered_model_files.append(model_files[model_index])
        model_files = filtered_model_files
    
    data = []

    for model_file in tqdm(model_files, desc='Models'):
        model_path = 'models/vgg_reconstructed_lcl/' + model_file

        vgg = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=0.2)
        model = VGGReconstructionLCL(vgg, learning_rate=config['learning_rate'], after_pooling=config['after_pooling'],
            num_multiplex=config['num_multiplex'], run_identifier='', lcl_distance=config['lcl_distance'],
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        model.load(model_path)
        model.features.lcl.enable()

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
    parser.add_argument('--run_ids', nargs='+', default=None)
    args = parser.parse_args()    

    check_mnist_c('vgg19r_lcl5_lr3e4', args.run_ids)


if __name__ == '__main__':
    main()