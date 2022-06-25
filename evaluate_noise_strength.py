import argparse
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from lateral_connections.character_models import TinyCNN 
from lateral_connections import MNISTCDataset
from lateral_connections.loaders import get_loaders
from lateral_connections.model_factory import load_model_by_key
from lateral_connections.layers import LaterallyConnectedLayer3

import wandb
import datetime


PRETRAINED_PATH = {
    'vgg19': 'models/vgg_with_lcl/VGG19_2022-04-04_183636__it13750_e2.pt',
    'tiny_cnn': 'models/tiny_cnn/TinyCNN_2022-06-15_215423__it20000_e4.pt'
}

NOISE_STRENGTHS = [0.0, 0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

def main(args):
    # Load Model and forward method *up to LCL*
    model = get_model(args.tiny)
    forward_fn = get_forward_fn(model, args.tiny)
    loader = get_loader(args.tiny)
    
    results = []

    for _ in tqdm(range(3), desc='Iterations'):
        # Run the check on the full training data
        eval_result = check_multiplex_changes(model, forward_fn, loader, args)

        print(f"\tRESULT FOR n={args.num_multiplex} and d={args.lcl_distance}:\t(tiny={args.tiny})")
        print('==================================================')
        for s in eval_result.keys():
            print(s, '\t', round(eval_result[s].item(),4))

        results.append(copy.deepcopy(eval_result))

    total_results = {}
    for i in range(len(results)):
        for s in results[i].keys():
            try:
                total_results[s].append(results[i][s].item())
            except KeyError:
                total_results[s] = [results[i][s].item()]

    # import code; code.interact(local=dict(globals(), **locals()))

    df = pd.DataFrame(total_results)
    means = df.mean().to_numpy()
    stds = df.std().to_numpy()

    plt.figure()
    plt.plot(eval_result.keys(), means)
    plt.fill_between(eval_result.keys(), means-stds, means+stds, alpha=0.2)
    plt.xscale('function', functions=(lambda x: x**(1/2), lambda x: x**2))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks([0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], rotation=90)
    plt.grid(True)
    plt.xlabel('Noise multiplier (eta)')
    plt.ylabel('Percentage of swapped multiplex cells')
    plt.tight_layout()
    model_type = 'tinyCNN' if args.tiny else 'VGG19'
    plt.savefig('noise_strength__' + model_type +'.png')
    plt.close()


def get_model(tiny=False):
    if tiny:
        # TinyCNN
        pre_model = load_model_by_key('tiny_cnn', model_path=PRETRAINED_PATH['tiny_cnn'])
        model = load_model_by_key('tiny_lateral_net')
        model.transfer_cnn_weights(pre_model)

    else:
        # VGG19
        model = load_model_by_key('vggonly', model_path=PRETRAINED_PATH['vgg19'])

    return model


def get_lcl(args):
    if args.tiny:
        lcl = LaterallyConnectedLayer3(args.num_multiplex, 10, 14, 14,
            d=args.lcl_distance, prd=args.lcl_distance,
            disabled=False, update=True,
            alpha=0.1, beta=0.001, gamma=0.001, eta=1, theta=0.0, iota=0.1, mu_batch_size=0, num_noisy_iterations=1000,
            use_scaling=True, random_k_change=False, random_multiplex_selection=False, gradient_learn_k=False)

    else:
        # lcl = LaterallyConnectedLayer3(args.num_multiplex, 512, 14, 14,
        lcl = LaterallyConnectedLayer3(args.num_multiplex, 128, 56, 56,
            d=args.lcl_distance, prd=args.lcl_distance,
            disabled=False, update=True,
            alpha=0.1, beta=0.001, gamma=0.001, eta=1, theta=0.0, iota=0.1, mu_batch_size=0, num_noisy_iterations=1000,
            use_scaling=True, random_k_change=False, random_multiplex_selection=False, gradient_learn_k=False)

    lcl.freeze()
    return lcl


def get_loader(tiny=False, batch_size=10):
    if tiny:
        def small_transform():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))
            ])
        dataset = MNIST('images/mnist/', train=True, transform=small_transform(), download=True)
        train_size = 50000
        eval_size = len(dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    else:
        train_loader, _, _, _ = get_loaders(batch_size=batch_size)

    return train_loader


def get_forward_fn(model, tiny=False):
    if tiny:
        def forward(x: torch.Tensor):
            x = model.conv(x)
            x = model.conv_act(x)
            x = model.maxpool(x)
            return x
    else:
        def forward(x: torch.Tensor):
            x = model.features.pool1(x)
            x = model.features.pool2(x)
            # x = model.features.pool3(x)
            # x = model.features.pool4(x)
            return x

    return forward


def check_with_noise(impact, lcl, noise_strength, batch_size=10):
    impact = torch.clone(impact)

    # Add noise
    impact += noise_strength * torch.normal(torch.zeros(impact.shape), torch.ones(impact.shape)).to(lcl.device)

    # Remove Self-dependences
    diagonal_repetition_mask = 1 - torch.eye(lcl.num_fm.item(), device=lcl.device).repeat(lcl.n, lcl.n)
    diagonal_repetition_mask += torch.eye(int(lcl.num_fm*lcl.n), device=lcl.device)
    impact *= diagonal_repetition_mask.unsqueeze(0)

    # Source Multiplex Select
    impact_reshaped = impact.reshape((batch_size, lcl.n, lcl.num_fm, lcl.n * lcl.num_fm))

    idx = torch.argmax(impact_reshaped, dim=1, keepdims=True)
    active_multiplex_source_mask = torch.zeros_like(impact_reshaped).scatter_(1, idx, 1.)
    active_multiplex_source_mask = active_multiplex_source_mask.reshape((batch_size, lcl.n*lcl.num_fm, lcl.n*lcl.num_fm))

    impact *= active_multiplex_source_mask

    # Target Multiplex Select
    impact_reshaped = impact.transpose(0,1).reshape((lcl.n*lcl.num_fm, batch_size, lcl.n, lcl.num_fm))
    impact_target_sum = torch.sum(impact_reshaped, dim=0)
    idx = torch.argmax(impact_target_sum, dim=1, keepdims=True)
    active_multiplex_target_mask = torch.zeros_like(impact_target_sum).scatter_(1, idx, 1.).reshape((batch_size, lcl.n*lcl.num_fm))
    active_multiplex_target_mask = active_multiplex_target_mask * impact.transpose(1,0)
    active_multiplex_target_mask = active_multiplex_target_mask.transpose(1,0)

    impact *= active_multiplex_target_mask

    # Check Selected
    indices = torch.where(impact > torch.Tensor([0]).unsqueeze(-1).unsqueeze(-1).to(lcl.device))
    selected = torch.zeros((batch_size, lcl.num_fm*lcl.n, lcl.num_fm*lcl.n))
    selected[indices] = 1
    return selected


def check_multiplex_changes(model, forward_fn, loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batches_to_run = max(128, args.min_images / loader.batch_size)

    model.to(device)
    lcl = get_lcl(args)
    lcl.to(device)

    evaluation = {}
    eval_result = {}
    for s in NOISE_STRENGTHS:
        evaluation[s] = 0
        eval_result[s] = 0

    for i, (images, labels) in tqdm(enumerate(loader, 0), total=batches_to_run, desc='Running over dataset...'):
        images = images.to(device)

        current_selected = {}

        with torch.no_grad():
            x = forward_fn(images)

            x = x.repeat(1, lcl.n, 1, 1)
            padX = lcl.pad_activations(x)

            impact = []
            for idx in range(padX.shape[1]):
                kernel = lcl.K[:, idx, ...].unsqueeze(1)
                lateral_impact = F.conv2d(padX, kernel, groups=int(lcl.n*lcl.num_fm), padding=0)
                impact.append(torch.sum(lateral_impact.unsqueeze(2), dim=(-2,-1)))
            impact = torch.cat(impact, dim=2)

            impact -= impact.min()
            if impact.max() != 0:
                impact /= impact.max()

            for s in evaluation.keys():
                selected = check_with_noise(impact, lcl, s, batch_size=loader.batch_size)
                current_selected[s] = (torch.sum(selected, dim=1) / selected.shape[1]).to(torch.bool).to(torch.int)
                
                changed = current_selected[s] - current_selected[0]
                changed[changed < 0] = 0
                evaluation[s] += torch.sum(changed)

        if i >= batches_to_run:
            break

    for s in evaluation.keys():
        eval_result[s] = evaluation[s] / loader.batch_size / int(batches_to_run) / lcl.num_fm.item()
    return eval_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiny', default=False, action='store_true', help='Whether to run on the large pre-trained VGG or the TinyCNN')
    parser.add_argument('--lcl_distance', default=1, type=int)
    parser.add_argument('--num_multiplex', default=4, type=int)
    parser.add_argument('--min_images', default=2000, type=int)
    args = parser.parse_args()
    main(args)
