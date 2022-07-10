import sys
import datetime
import logging
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from lateral_connections.model_factory import load_model_by_key, get_config_by_key
from lateral_connections.loaders import *
from lateral_connections.loaders import get_loaders, load_mnistc

import wandb


def small_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])


def get_mnistc_loader(variant):
    dataset = load_mnistc(variant, transform=small_transform())
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)
    

def check_mnist_c(model, run_config):
    wandb_run_name = run_config['wandb_run_name']

    mnist_c_variants = [
        'identity', 'pixelate', 'dotted_line', 'gaussian_blur', 'elastic_transform', 'jpeg_compression', 'speckle_noise',
        'glass_blur', 'spatter', 'translate', 'fog', 'shear', 'scale', 'zigzag', 'defocus_blur', 'gaussian_noise',
        'contrast', 'canny_edges', 'zoom_blur', 'line', 'pessimal_noise', 'rotate', 'brightness', 'shot_noise',
        'saturate', 'motion_blur', 'snow', 'inverse', 'impulse_noise', 'stripe', 'quantize', 'frost'
    ]
    data = []

    model.eval()

    for variant in tqdm(mnist_c_variants, desc='MNIST-C Variants', leave=False):
        loader = get_mnistc_loader(variant)
        c_acc, c_loss = model.test(loader)
        
        print(f"{wandb_run_name.ljust(45)}\t{variant.ljust(20)}\t{round(c_acc,4)}")
        data.append({
            'model': wandb_run_name,
            'mnist_c_variant': variant,
            'accuracy': c_acc
        })

    df = pd.DataFrame(data)
    df['model_type'] = run_config['model_type']
    df.to_csv(wandb_run_name + '__results.csv', index=False)

def eval_run(run_config):
    config = get_config_by_key('tiny_cnn')
    model = load_model_by_key('tiny_cnn', config=config)

    wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
    wandb.init(
        project='MT_LateralConnections',
        entity='lehl',
        group='eval',
        name='xxx',
        config=config,
        mode='disabled',
        reinit=True
    )

    model.load('models/tiny_cnn/' + run_config['wandb_run_name'] + '__best.pt')
    check_mnist_c(model, run_config)

def main():
    runs = [
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_125523'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_130352'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_124048'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_130127'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_125013'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_122944'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_124331'},
        {'model_type': 'TinyCNN_FullyTrained', 'wandb_run_name': '2022-07-01_123001'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_212353'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_213210'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_213158'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_213244'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_212345'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_212352'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_212945'},
        {'model_type': 'TinyCNN_PreTrained', 'wandb_run_name': '2022-07-04_212354'},
    ]

    for run_config in runs:
        eval_run(run_config)

if __name__ == '__main__':
    main()