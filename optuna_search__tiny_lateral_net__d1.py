import argparse
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


def get_dataset_loaders():
    def small_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])

    train_dataset = MNIST('images/mnist/', train=True, transform=small_transform(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_dataset = MNIST('images/mnist/', train=False, transform=small_transform(), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    return train_loader, test_loader

def optuna_objective(trial):
    config = get_config_by_key('tiny_lateral_net')
    config['lcl_distance'] = 1

    config['lcl_eta'] = trial.suggest_float('eta', 1e-3, 5e-1, log=True)
    config['lcl_alpha'] = trial.suggest_float('alpha', 1e-4, 1e-0, log=True)
    config['num_multiplex'] = trial.suggest_int('n', 2, 5)
    config['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    # pretrained_cnn = load_model_by_key('tiny_cnn')
    # raise ArgumentError('Missing Path')
    # pretrained_cnn.load('TODO: PATH')

    model = load_model_by_key('tiny_lateral_net', config=config)
    # model.transfer_cnn_weights()

    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wandb_run_name = base_name
    wandb_group_name = 'TinyLateralNet_Optuna__FreshlyTrained__d1'

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
    eval_score = model.train_optuna(train_loader, test_loader, trial, num_epochs=16)
    
    return eval_score


def main():
    study_name = 'tiny_lateral_net__optuna__full_trained__d1'
    storage_name = "sqlite:///{}.db".format(study_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')
    study.optimize(optuna_objective, n_trials=100)


if __name__ == '__main__':
    main()
