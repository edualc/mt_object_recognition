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
    config = get_config_by_key('tiny_cnn')
    config['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    model = load_model_by_key('tiny_cnn', config=config)

    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wandb_run_name = base_name
    wandb_group_name = 'TinyCNN_Optuna__FreshlyTrained'

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
    study_name = 'tiny_cnn__optuna__full_trained'
    storage_name = "sqlite:///{}.db".format(study_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')
    study.optimize(optuna_objective, n_trials=100)


if __name__ == '__main__':
    main()
