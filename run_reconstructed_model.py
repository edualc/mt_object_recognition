import argparse

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from lateral_connections import LateralModel, VggModel
from lateral_connections import VggWithLCL
from lateral_connections import MNISTCDataset
from lateral_connections.loaders import get_loaders
from lateral_connections.character_models import VGGReconstructionLCL

import wandb
import datetime

DO_WANDB = True

def main(args):
    config = {
        'num_classes': 10,
        'learning_rate': args.lr,
        'num_multiplex': args.num_multiplex,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lcl_alpha': args.lcl_alpha,
        'lcl_eta': args.lcl_eta,
        'lcl_theta': args.lcl_theta,
        'lcl_iota': args.lcl_iota,
        'lcl_distance': args.lcl_distance,
        'lcl_k': 2*args.lcl_distance+1,
        'mnistc_corruption': 'gaussian_noise',
        'model_path': args.model_path,
        'after_pooling': args.after_pooling,

        'use_scaling': args.use_scaling,
        'random_multiplex_selection': args.random_multiplex_selection,
        'gradient_learn_k': args.gradient_learn_k,
        'random_k_change': args.random_k_change,
        'fc_only': args.fc_only,

        'no_pretrained_vgg': args.no_pretrained_vgg,
    }

    train_network(config)

def train_network(config):
    base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    wandb_run_name = base_name + '_LCL' + str(args.after_pooling) + '_d' + str(args.lcl_distance)
    wandb_group_name = 'VGG19R_Scaling'

    if config['use_scaling']:
        wandb_group_name += '__use_scaling'

    elif config['random_k_change']:
        wandb_group_name += '__random_K_change'

    elif config['random_multiplex_selection']:
        wandb_group_name += '__random_multiplex_selection'

    elif config['gradient_learn_k']:
        wandb_group_name += '__gradient_learned_K'

    elif config['fc_only']:
        wandb_group_name += '__fc_only__AvgPool'

    elif config['no_pretrained_vgg']:
        wandb_group_name += '__no_pretrained_vgg'


    if DO_WANDB:
        wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
        wandb.init(
            project='MT_LateralConnections',
            entity='lehl',
            group=wandb_group_name,
            # group='debug',
            name=wandb_run_name,
            config=config,
            mode='disabled',
        )

    vgg = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=0.2)
    if not config['no_pretrained_vgg']:
        vgg.load(config['model_path'])

    model = VGGReconstructionLCL(vgg, learning_rate=config['learning_rate'], after_pooling=config['after_pooling'],
        num_multiplex=config['num_multiplex'], run_identifier=wandb_run_name, lcl_distance=config['lcl_distance'],
        lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'],
        random_k_change=config['random_k_change'],
        random_multiplex_selection=config['random_multiplex_selection'],
        gradient_learn_k=config['gradient_learn_k'],
        fc_only=config['fc_only'],
        freeze_vgg=(not config['no_pretrained_vgg']))
    # import code; code.interact(local=dict(globals(), **locals()))
    del vgg
    print(model)

    train_loader, val_loader, test_loader, corrupt_loader = get_loaders(config['batch_size'], corruption=config['mnistc_corruption'])

    model.train_with_loader(train_loader, val_loader, test_loader=corrupt_loader, num_epochs=config['num_epochs'])

    c_acc, c_loss = model.test(corrupt_loader)
    print(f"[VGG19_Reconstructed] MNIST-C '{config['mnistc_corruption']}':\t\tAccuracy:{c_acc:1.4f}\tLoss:{c_loss:1.4f}")

    # import code; code.interact(local=dict(globals(), **locals()))
    wandb.finish(quiet=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate of VGG19\'s optimizer')
    parser.add_argument('--num_multiplex', type=int, default=4, help='Number of multiplex cells in LCL layers')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs trained')
    parser.add_argument('--lcl_distance', type=int, default=2, help='Size of kernel filters for K, as k=2*d+1 with [k x k] sized filters')
    parser.add_argument('--lcl_alpha', type=float, default=1e-3, help='Rate at which kernel K is changed by K_change')
    parser.add_argument('--lcl_eta', type=float, default=1, help='Rate at which the output is changed by O=(1-eta)*A+eta*L')
    parser.add_argument('--lcl_theta', type=float, default=0.0, help='How much the noise is added to the LCL training (breaking symmetry)')
    parser.add_argument('--lcl_iota', type=float, default=0.5, help='Rate at which the argmax((1-iota)*A+iota*L) is calculated to determine active multiplex cells')
    # WandB: https://wandb.ai/lehl/MT_LateralConnections/runs/1d4yqrn9?workspace=user-lehl
    parser.add_argument('--model_path', type=str, default='models/vgg_with_lcl/VGG19_2022-04-04_183636__it13750_e2.pt', help='Vgg19 pretrained model to use')
    parser.add_argument('--after_pooling', type=int, default=5, help='after which pooling block the LCL is placed (1-5)')

    parser.add_argument('--use_scaling', default=True, action='store_true', help='Whether ')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of models to train')
    parser.add_argument('--random_k_change', default=False, action='store_true', help='Ablation study: use random values for K_change')
    parser.add_argument('--random_multiplex_selection', default=False, action='store_true', help='Ablation study: choose multiplex cells randomly')
    parser.add_argument('--gradient_learn_k', default=False, action='store_true', help='Ablation study: do not use hebbian learning but regular gradients to learn the LCL kernel')
    parser.add_argument('--fc_only', default=False, action='store_true', help='Ablation study: reconstruct VGG19 without LCL, only add new fully connected layer')

    parser.add_argument('--no_pretrained_vgg', default=False, action='store_true', help='Whether the pre-trained VGG should be used as a starting point')

    args = parser.parse_args()

    # import code; code.interact(local=dict(globals(), **locals()))
    for _ in range(args.num_runs):
        main(args)


