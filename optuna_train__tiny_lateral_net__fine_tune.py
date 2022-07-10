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
import optuna


def small_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])


def get_dataset_loaders():
    train_dataset = MNIST('images/mnist/', train=True, transform=small_transform(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_dataset = MNIST('images/mnist/', train=False, transform=small_transform(), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    return train_loader, test_loader


def get_mnistc_loader(variant):
    dataset = load_mnistc(variant, transform=small_transform())
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)
    

def check_mnist_c(model, wandb_run_name):
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
    df['model_type'] = 'TinyLateralNet__Optuna_BestRun__FineTune2'
    df.to_csv(wandb_run_name + '__results.csv', index=False)
    

def train_model(study):
    best_trial = study.best_trial

    config = get_config_by_key('tiny_lateral_net')
    # from WandB Parallel Plot
    # see https://wandb.ai/lehl/MT_LateralConnections/groups/TinyLateralNet_Optuna__ConvPreTrained
    #
    config['lcl_alpha'] = 0.05
    config['num_multiplex'] = 3
    config['lcl_distance'] = 0
    config['lcl_eta'] = 0.25
    config['learning_rate'] = 0.0002

    pretrained_model = load_model_by_key('tiny_cnn')
    pretrained_model.load('models/tiny_cnn/2022-07-01_124048__ep4.pt')

    model = load_model_by_key('tiny_lateral_net', config=config)
    model.transfer_cnn_weights(pretrained_model)
    del pretrained_model

    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wandb_run_name = base_name
    wandb_group_name = 'TinyLateralNet_Optuna__FinalRun__ConvPreTrained2'

    model.run_identifier = wandb_run_name

    wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
    wandb.init(
        project='MT_LateralConnections',
        entity='lehl',
        group=wandb_group_name,
        #group='debug',
        name=wandb_run_name,
        config=config,
        # mode='disabled',
        reinit=True
    )

    train_loader, test_loader = get_dataset_loaders()
    model.train_with_loader(train_loader, test_loader, num_epochs=16)

    check_mnist_c(model, wandb_run_name)


def main():
    study_name = 'tiny_lateral_net__optuna__conv_pre_trained'
    storage_name = "sqlite:///{}.db".format(study_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')

    for _ in range(2):
        train_model(study)


if __name__ == '__main__':
    main()