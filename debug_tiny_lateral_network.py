import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import OrderedDict

from lateral_connections import LateralModel, VggModel
from lateral_connections import VggWithLCL
from lateral_connections import MNISTCDataset
from lateral_connections.loaders import get_loaders, load_mnistc
from lateral_connections.layers import LaterallyConnectedLayer, LaterallyConnectedLayer2, LaterallyConnectedLayer3
from lateral_connections.torch_utils import *

import wandb
import datetime

DO_WANDB = True

config = {
    'num_classes': 10,
    'learning_rate': 3e-5,
    'num_multiplex': 4,
    'batch_size': 4,
    'num_epochs': 20,
    'lcl_alpha': 1e-2,
    'lcl_eta': 0.1,
    'lcl_theta': 0.2,
    'lcl_iota': 0.2,
    'lcl_distance': 1,
    'conv_size': 5,
    'use_scaling': True,
}

base_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
wandb_run_name = base_name
wandb_group_name = 'TinyLCL_Debug'

wandb.login(key='efd0a05b7bd26ed445bf073625a373b845fc9385')
wandb.init(
    project='MT_LateralConnections',
    entity='lehl',
    #group=wandb_group_name,
    group='debug',
    name=wandb_run_name,
    config=config,
    # mode='disabled',
)

def plot_kernels(model, plot_scale=3):
    num_kernels = model.lcl.K.shape[0]

    fig, axs = plt.subplots(num_kernels, num_kernels, figsize=(plot_scale*num_kernels, plot_scale*num_kernels))

    kernel_data = model.lcl.K.cpu()

    for x in range(num_kernels):
        for y in range(num_kernels):
            axs[x,y].imshow(kernel_data[x,y,...])
            
    plt.show()
    plt.close()

class TinyLateralNetwork(nn.Module):
    def __init__(self, config):
        super(TinyLateralNetwork, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=config['conv_size'], padding=1, kernel_size=(3,3))
        self.act1 = nn.Sigmoid()
        self.maxpool = nn.AdaptiveMaxPool2d((14, 14))
        self.lcl = LaterallyConnectedLayer3(self.config['num_multiplex'], config['conv_size'], 14, 14,
                              d=self.config['lcl_distance'],
                              prd=self.config['lcl_distance'],
                              disabled=False,
                              alpha=self.config['lcl_alpha'],
                              beta=(1 / ((50000/2) / self.config['batch_size'])),
                              eta=self.config['lcl_eta'],
                              theta=self.config['lcl_theta'],
                              iota=self.config['lcl_iota'],
                              use_scaling=config['use_scaling'],
                              random_k_change=False,
                              random_multiplex_selection=False,
                              gradient_learn_k=False)
        # self.lcl = LaterallyConnectedLayer2(self.config['num_multiplex'], config['conv_size'], 14, 14,
        #                       d=self.config['lcl_distance'],
        #                       prd=self.config['lcl_distance'],
        #                       disabled=False,
        #                       alpha=self.config['lcl_alpha'],
        #                       eta=self.config['lcl_eta'],
        #                       theta=self.config['lcl_theta'],
        #                       iota=self.config['lcl_iota'],
        #                       use_scaling=config['use_scaling'],
        #                       random_k_change=False,
        #                       random_multiplex_selection=False,
        #                       gradient_learn_k=False)
        
        self.fc1 = nn.Linear(in_features=config['conv_size']*14*14, out_features=100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=self.config['num_classes'])
        
        self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=self.config['learning_rate'], momentum=0.9, weight_decay=0.0005)
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.lcl(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x
    
    def train_with_loader(self, train_loader, val_loader, test_loader=None, num_epochs=5):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[INFO]: {total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO]: {total_trainable_params:,} trainable parameters.")

        for epoch in range(num_epochs):
            print(f"[INFO]: Epoch {epoch} of {num_epochs}")
            self.train()

            correct = 0
            total = 0
            total_loss = 0
            counter = 0

            agg_correct = 0
            agg_total = 0
            agg_loss = 0

            # Training Loop
            for i, (images, labels) in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='Training'):
                counter += 1
                
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                loss = self.loss_fn(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
        
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += (loss.item() / labels.size(0))

                current_iteration = epoch*len(train_loader) + i

                if current_iteration > 0 and (current_iteration % 1250) == 0:
                    #self.save(f"models/vgg_with_lcl/{self.run_identifier}__it{current_iteration}_e{epoch}.pt")
                    val_acc, val_loss = self.test(val_loader)
                    self.train()
                    log_dict = { 'val_loss': val_loss, 'val_acc': val_acc, 'iteration': current_iteration }

                    if test_loader:
                        test_acc, test_loss = self.test(test_loader)
                        log_dict['test_acc'] = test_acc
                        log_dict['test_loss'] = test_loss

                    wandb.log(log_dict, commit=False)

                if (current_iteration % 250) == 0:
                    wandb.log({ 'train_batch_loss': round(total_loss/250,4), 'train_batch_acc': round(correct/total,4), 'iteration': current_iteration })
                    
                    agg_correct += correct
                    agg_total += total
                    agg_loss += total_loss

                    correct = 0
                    total = 0
                    total_loss = 0
                    counter = 0
                
            plot_kernels(self)
            wandb.log({'epoch': epoch, 'iteration': current_iteration, 'train_loss': agg_loss/len(train_loader), 'train_acc': agg_correct/agg_total})

    def test(self, test_loader):
        self.eval()

        correct = 0
        total = 0
        total_loss = 0
        counter = 0

        # Evaluation Loop
        for i, (images, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='Testing', leave=False):
            counter += 1
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self(images)
            loss = self.loss_fn(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
    
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += (loss.item() / labels.size(0))

        total_loss /= counter
        acc = correct / total

        return acc, total_loss

model = TinyLateralNetwork(config)
model.to(model.device)

def small_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

dataset = MNIST('images/mnist/', train=True, transform=small_transform(), download=True)

train_size = 50000
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)

corrupt_dataset = load_mnistc('line')
corrupt_loader = torch.utils.data.DataLoader(corrupt_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)


model.train_with_loader(train_loader, val_loader, num_epochs=config['num_epochs'])
