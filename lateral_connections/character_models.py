from typing import Union, List, Dict, Any, cast
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

import torchvision
import torchvision.transforms as transforms
from torchvision.models.vgg import VGG

import wandb

from .layers import LaterallyConnectedLayer


# Taken from https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg19
#
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# Taken from https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg19
#
# Changed to have only 1 in_channel.
#
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def build_custom_vgg19(dropout, num_classes, config_ident='E'):
    net = VGG(make_layers(cfgs[config_ident], batch_norm=False))
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(4096, num_classes)
        # nn.Softmax(-1)
    )
    return net

class VggWithLCL(nn.Module):
    def __init__(self, num_classes, learning_rate, dropout, num_multiplex=4, do_wandb=False, run_identifier="",
        lcl_alpha=0.1, lcl_eta=0.1, lcl_theta=0.1, lcl_iota=0.1):

        super(VggWithLCL, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_multiplex = num_multiplex
        self.do_wandb = do_wandb
        self.run_identifier = run_identifier

        self.lcl_alpha = lcl_alpha
        self.lcl_theta = lcl_theta
        self.lcl_eta = lcl_eta
        self.lcl_iota = lcl_iota

        self.net = build_custom_vgg19(dropout=self.dropout, num_classes=self.num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._add_lcl_to_network()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)

    def _add_lcl_to_network(self):
        self.features = nn.Sequential(
            OrderedDict([
                ('pool1', self.net.features[:5]),
                ('lcl1',  LaterallyConnectedLayer(self.num_multiplex, 64, 112, 112, d=2, 
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool2', self.net.features[5:10]),
                ('lcl2',  LaterallyConnectedLayer(self.num_multiplex, 128, 56, 56, d=2, 
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool3', self.net.features[10:19]),
                ('lcl3',  LaterallyConnectedLayer(self.num_multiplex, 256, 28, 28, d=2, 
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool4', self.net.features[19:28]),
                ('lcl4',  LaterallyConnectedLayer(self.num_multiplex, 512, 14, 14, d=2, 
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool5', self.net.features[28:]),
            ])
        )
        self.avgpool = self.net.avgpool
        self.classifier = self.net.classifier
        self.to(self.device)
        del self.net

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
                    self.save(f"models/vgg_with_lcl/{self.run_identifier}__it{current_iteration}_e{epoch}.pt")
                    val_acc, val_loss = self.test(val_loader)
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

            wandb.log({'epoch': epoch, 'iteration': current_iteration, 'train_loss': agg_loss/len(train_loader), 'train_acc': agg_correct/agg_total})

    def test(self, test_loader):
        self.eval()

        correct = 0
        total = 0
        total_loss = 0
        counter = 0

        # Evaluation Loop
        for i, (images, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='Testing'):
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

class SmallVggWithLCL(VggWithLCL):
    def __init__(self, num_classes, learning_rate, dropout, num_multiplex=4, do_wandb=False, run_identifier="",
        lcl_alpha=0.1, lcl_eta=0.1, lcl_theta=0.1, lcl_iota=0.1):

        super(VggWithLCL, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_multiplex = num_multiplex
        self.do_wandb = do_wandb
        self.run_identifier = run_identifier

        self.lcl_alpha = lcl_alpha
        self.lcl_theta = lcl_theta
        self.lcl_eta = lcl_eta
        self.lcl_iota = lcl_iota

        self.net = build_custom_vgg19(dropout=self.dropout, num_classes=self.num_classes, config_ident='D')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._add_lcl_to_network()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)

    def _add_lcl_to_network(self):
        self.features = nn.Sequential(
            OrderedDict([
                ('pool1', self.net.features[:5]),
                # ('lcl1',  LaterallyConnectedLayer(self.num_multiplex, 64, 112, 112, d=2, 
                #     alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool2', self.net.features[5:10]),
                # ('lcl2',  LaterallyConnectedLayer(self.num_multiplex, 128, 56, 56, d=2, 
                #     alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool3', self.net.features[10:17]),
                ('lcl3',  LaterallyConnectedLayer(self.num_multiplex, 256, 28, 28, d=2, disabled=False, 
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool4', self.net.features[17:24]),
                # ('lcl4',  LaterallyConnectedLayer(self.num_multiplex, 512, 14, 14, d=2, 
                #     alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota)),
                ('pool5', self.net.features[24:]),
            ])
        )
        self.avgpool = self.net.avgpool
        self.classifier = self.net.classifier
        self.to(self.device)
        del self.net


class VGGReconstructionLCL(nn.Module):
    def __init__(self, vgg, after_pooling=3, learning_rate=3e-4, num_multiplex=4, run_identifier="",
        lcl_distance=2, lcl_alpha=1e-3, lcl_eta=0.0, lcl_theta=0.2, lcl_iota=0.2,
        use_scaling=False, random_k_change=False, random_multiplex_selection=False, gradient_learn_k=False, fc_only=False, freeze_vgg=True):

        super(VGGReconstructionLCL, self).__init__()

        self.vgg = vgg
        self.num_classes = self.vgg.num_classes
        self.device = self.vgg.device

        self.run_identifier = run_identifier

        self.after_pooling = after_pooling
        self.learning_rate = learning_rate

        self.num_multiplex = num_multiplex
        self.lcl_distance = lcl_distance
        self.lcl_alpha = lcl_alpha
        self.lcl_theta = lcl_theta
        self.lcl_eta = lcl_eta
        self.lcl_iota = lcl_iota

        self.freeze_vgg = freeze_vgg
        self.use_scaling = use_scaling

        # Ablation study variants
        self.random_k_change = random_k_change
        self.random_multiplex_selection = random_multiplex_selection
        self.gradient_learn_k = gradient_learn_k
        self.fc_only = fc_only


        self._reconstruct_from_vgg()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)

    # lehl@2022-04-17: Replacement of Convnet/FC layers after LCL through only new FC layers,
    # where the number of parameters is changed such that the majority of parameters lie in the LCL
    # (and thus roughly equal numbers of parameters are used between VGG19 and the reconstruction)
    #
    def _reconstruct_from_vgg(self):
        pooling_sizes = {
            'channels': { 1: 64, 2: 128, 3: 256, 4: 512, 5: 512 },
            'width': { 1: 56, 2: 28, 3: 14, 4: 7, 5: 7 },
            'height': { 1: 56, 2: 28, 3: 14, 4: 7, 5: 7 },
        }

        # The LCL kernel is proportional sized to the number of feature maps/channels (C),
        # the number of multiplex cells (M) and the size of the filter kernels (k) - and
        # to all of those three parameters *squared*
        #
        def lcl_num_parameters(pooling_layer):
            C = pooling_sizes['channels'][pooling_layer]
            M = self.num_multiplex
            k = 2 * self.lcl_distance + 1
            return C**2 * M**2 * k**2

        def sizes_for_pooling_layer(pooling_layer):
            nC = pooling_sizes['channels'][pooling_layer]
            nH = pooling_sizes['height'][pooling_layer]
            nW = pooling_sizes['width'][pooling_layer]
            return nH, nW, nC

        def fc_input_size(pooling_layer):
            nH, nW, nC = sizes_for_pooling_layer(pooling_layer)
            return nH * nW * nC

        def fc_size(pooling_layer, num_vgg_params=139610058):
            num_lcl_params = lcl_num_parameters(pooling_layer)
            params_remaining = num_vgg_params - num_lcl_params
            divisor = (10 + (fc_input_size(pooling_layer)))
            result = params_remaining // divisor

            if result <= 0:
                raise ArgumentError(f'LCL: Not enough parameters remaining for FC layers: #LCL: {num_lcl_params}, remaining: {params_remaining}')

            return result

        if self.after_pooling == 1:
            vgg19_unit = nn.Sequential(*(list(self.vgg.features.pool1)))
            lcl_layer = LaterallyConnectedLayer(self.num_multiplex, 64, 112, 112, d=self.lcl_distance, prd=self.lcl_distance, disabled=False,
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota,
                    use_scaling=self.use_scaling, random_k_change=self.random_k_change, random_multiplex_selection=self.random_multiplex_selection, gradient_learn_k=self.gradient_learn_k)

        elif self.after_pooling == 2:
            vgg19_unit = nn.Sequential(*(list(self.vgg.features.pool1) + list(self.vgg.features.pool2)))
            lcl_layer = LaterallyConnectedLayer(self.num_multiplex, 128, 56, 56, d=self.lcl_distance, prd=self.lcl_distance, disabled=False,
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota,
                    use_scaling=self.use_scaling, random_k_change=self.random_k_change, random_multiplex_selection=self.random_multiplex_selection, gradient_learn_k=self.gradient_learn_k)

        elif self.after_pooling == 3:
            vgg19_unit = nn.Sequential(*(list(self.vgg.features.pool1) + list(self.vgg.features.pool2) + list(self.vgg.features.pool3)))
            lcl_layer = LaterallyConnectedLayer(self.num_multiplex, 256, 28, 28, d=self.lcl_distance, prd=self.lcl_distance, disabled=False,
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota,
                    use_scaling=self.use_scaling, random_k_change=self.random_k_change, random_multiplex_selection=self.random_multiplex_selection, gradient_learn_k=self.gradient_learn_k)

        elif self.after_pooling == 4:
            vgg19_unit = nn.Sequential(*(list(self.vgg.features.pool1) + list(self.vgg.features.pool2) + list(self.vgg.features.pool3) + list(self.vgg.features.pool4)))
            lcl_layer = LaterallyConnectedLayer(self.num_multiplex, 512, 14, 14, d=self.lcl_distance, prd=self.lcl_distance, disabled=False,
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota,
                    use_scaling=self.use_scaling, random_k_change=self.random_k_change, random_multiplex_selection=self.random_multiplex_selection, gradient_learn_k=self.gradient_learn_k)

        elif self.after_pooling == 5:
            vgg19_unit = nn.Sequential(*(list(self.vgg.features.pool1) + list(self.vgg.features.pool2) + list(self.vgg.features.pool3) + list(self.vgg.features.pool4) + list(self.vgg.features.pool5)))
            lcl_layer = LaterallyConnectedLayer(self.num_multiplex, 512, 7, 7, d=self.lcl_distance, prd=self.lcl_distance, disabled=False,
                    alpha=self.lcl_alpha, eta=self.lcl_eta, theta=self.lcl_theta, iota=self.lcl_iota,
                    use_scaling=self.use_scaling, random_k_change=self.random_k_change, random_multiplex_selection=self.random_multiplex_selection, gradient_learn_k=self.gradient_learn_k)

        if self.fc_only:
            vgg19_unit = nn.Sequential(*(list(self.vgg.features.pool1) + list(self.vgg.features.pool2) + list(self.vgg.features.pool3) + list(self.vgg.features.pool4) + list(self.vgg.features.pool5)))
            self.features = nn.Sequential( OrderedDict([ ('vgg19_unit', vgg19_unit) ]) )
        else:
            self.features = nn.Sequential( OrderedDict([ ('vgg19_unit', vgg19_unit), ('lcl', lcl_layer) ]) )

        # Freeze the params of the previous layers of VGG19
        #
        if self.freeze_vgg:
            for param in self.features.vgg19_unit.parameters():
                param.requires_grad = False

        if self.fc_only:
            self.maxpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Linear(4096, self.num_classes)
            )

        else:
            fc_layer_input_size = fc_input_size(self.after_pooling)
            fc_layer_output_size = fc_size(self.after_pooling)
            nH, nW, _ = sizes_for_pooling_layer(self.after_pooling)
            self.maxpool = nn.AdaptiveMaxPool2d((nH, nW))
            self.classifier = nn.Sequential(
                nn.Linear(fc_layer_input_size, fc_layer_output_size),
                nn.ReLU(True),
                nn.Linear(fc_layer_output_size, self.num_classes)
            )

        # Delete the pretrained "full" VGG19, since we're only
        # interested in using it a pretraining for earlier layers
        #
        del self.vgg
        self.to(self.device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_with_loader(self, train_loader, val_loader, test_loader=None, num_epochs=5):
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
                    self.save(f"models/vgg_reconstructed_lcl/{self.run_identifier}__it{current_iteration}_e{epoch}.pt")
                    val_acc, val_loss = self.test(val_loader)
                    log_dict = { 'val_loss': val_loss, 'val_acc': val_acc, 'iteration': current_iteration }

                    if test_loader:
                        test_acc, test_loss = self.test(test_loader)
                        log_dict['test_acc'] = test_acc
                        log_dict['test_loss'] = test_loss

                    wandb.log(log_dict, commit=False)

                if (current_iteration % 250) == 0:
                    if i > 0:
                        wandb.log({ 'train_batch_loss': round(total_loss/250,4), 'train_batch_acc': round(correct/total,4), 'iteration': current_iteration })
                    
                    agg_correct += correct
                    agg_total += total
                    agg_loss += total_loss

                    correct = 0
                    total = 0
                    total_loss = 0
                    counter = 0

            wandb.log({'epoch': epoch, 'iteration': current_iteration, 'train_loss': agg_loss/len(train_loader), 'train_acc': agg_correct/agg_total})

    def test(self, test_loader):
        self.eval()

        correct = 0
        total = 0
        total_loss = 0
        counter = 0

        # Evaluation Loop
        for i, (images, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='Testing'):
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


# GPU Performance Blogpost:
# https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
#
class CharacterVgg():
    def __init__(self, dataset_path, num_classes, batch_size, learning_rate, dropout):
        self.num_classes = num_classes

        self.net = build_custom_vgg19(dropout=dropout, num_classes=num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])

        self._setup_dataset(dataset_path)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def _setup_dataset(self, dataset_path):
        self.train_dataset = None
        self.test_dataset = None
        self.dataset_name = ''
        raise NotImplementedError('Please implement the :_setup_dataset method.')

    def save_model(self, epoch):
        torch.save(self.net.state_dict(), f'models/character_vgg/{self.dataset_name}_ep{str(epoch).zfill(3)}.pt')

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def train(self, num_epochs=25):
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"[INFO]: {total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"[INFO]: {total_trainable_params:,} trainable parameters.")

        for epoch in range(num_epochs):
            print(f"[INFO]: Epoch {epoch} of {num_epochs}")

            self.net.train()

            correct = 0
            total = 0
            total_loss = 0
            counter = 0

            # Training Loop
            for i, (images, labels) in tqdm(enumerate(self.train_loader, 0), total=len(self.train_loader), desc='Training'):
                counter += 1
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(images)
                loss = self.loss_fn(outputs, labels)
                
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
        
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += (loss.item() / labels.size(0))

            total_loss /= counter
            acc = correct / total

            # Validation Loop
            val_acc, val_loss = self.test()

            log_msg = ''.join([
                "Epoch",       f"{epoch+1:2d}",                 "\t",
                "loss:",       f"{round(total_loss, 4):1.4f}",  "\t",
                "acc:",        f"{round(acc, 4):1.4f}",         "\t",
                "val_loss:",   f"{round(val_loss, 4):1.4f}",    "\t",
                "val_acc:",    f"{round(val_acc, 4):1.4f}"
            ])
            print(log_msg)
            wandb.log({ 'loss': total_loss, 'acc': acc, 'val_loss': val_loss, 'val_acc': val_acc })
            self.save_model(epoch)

    def test(self):
        self.net.eval()
    
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))

        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            counter = 0

            for i, (images, labels) in tqdm(enumerate(self.test_loader, 0), total=len(self.test_loader), desc='Validation'):
                counter += 1
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, preds = torch.max(outputs, 1)
                
                loss = self.loss_fn(outputs, labels)

                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

                correct_batch = (preds == labels).squeeze()
                for k in range(len(preds)):
                    label = labels[k]
                    class_correct[label] += correct_batch[k].item()
                    class_total[label] += 1
        
            total_loss /= counter
            acc = correct/total

            print('')
            for k in range(self.num_classes):
                print(f"Accuracy of digit {k}: {100*class_correct[k]/class_total[k]}")
            print('')

            return acc, total_loss

class MNISTVgg(CharacterVgg):
    def __init__(self, dataset_path, num_classes=10, batch_size=32, learning_rate=0.003, dropout=0.2):
        super().__init__(dataset_path, num_classes, batch_size, learning_rate, dropout)

    def _setup_dataset(self, dataset_path):
        self.train_dataset = torchvision.datasets.MNIST(dataset_path, train=True, transform=self.train_transform, download=True)
        self.test_dataset = torchvision.datasets.MNIST(dataset_path, train=False, transform=self.test_transform, download=True)
        self.dataset_name = 'MNIST'

class OmniglotVgg(CharacterVgg):
    def __init__(self, dataset_path, num_classes=1623, batch_size=32, learning_rate=0.003, dropout=0.2):
        super().__init__(dataset_path, num_classes, batch_size, learning_rate, dropout)

    def _setup_dataset(self, dataset_path):
        self.train_dataset = torchvision.datasets.Omniglot(dataset_path, background=True, transform=self.train_transform, download=True)
        self.test_dataset = torchvision.datasets.Omniglot(dataset_path, background=False, transform=self.test_transform, download=True)
        self.dataset_name = 'Omniglot'

if __name__ == '__main__':

    # wandb_run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # wandb.init(project="CharacterVGG", entity="lehl", group='Omniglot', name=wandb_run_name)

    m = OmniglotVgg('images/omniglot/')

    train_labels = []
    test_labels = []

    for img, lbl in m.train_dataset:
        train_labels.append(lbl)

    for img, lbl in m.test_dataset:
        test_labels.append(lbl)
    # import code; code.interact(local=dict(globals(), **locals()))



    # wandb.watch(m.net)

    # m.train(5)
