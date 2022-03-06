from typing import Union, List, Dict, Any, cast
from PIL import Image
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
from torch import optim

import torchvision
import torchvision.transforms as transforms
from torchvision.models.vgg import VGG

import wandb


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

def build_custom_vgg19(dropout, num_classes):
    net = VGG(make_layers(cfgs["E"], batch_norm=False))
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
    import code; code.interact(local=dict(globals(), **locals()))



    # wandb.watch(m.net)

    # m.train(5)