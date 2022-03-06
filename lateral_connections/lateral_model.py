import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from .character_models import MNISTVgg, OmniglotVgg
from .dataset import CustomImageDataset
from .torch_utils import *
from .vgg_model import VggModel

from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import datetime
import h5py
import copy



pretrained_models = {
    'MNIST': { 'class': MNISTVgg, 'path': 'models/character_vgg/MNIST_pretrained.pt', 'dataset_path': 'images/mnist/' },
    # 'Omniglot': { 'class': OmniglotVgg, 'path': None, 'dataset_path': 'images/omniglot/' }
}

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class LaterallyConnectedLayer(nn.Module):
    def __init__(self, n, num_fm, fm_height, fm_width):
        super().__init__()
        self.n = n
        self.num_fm = num_fm
        self.fm_height = fm_height
        self.fm_width = fm_width


    def forward(self, x):
        return x


class VggLateral(nn.Module):
    def __init__(self, dataset_name, num_crosstalk_replications=4):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_crosstalk_replications = num_crosstalk_replications

        char_model_class = pretrained_models[dataset_name]['class']
        dataset_path = pretrained_models[dataset_name]['dataset_path']
        self.pretrained_vgg = char_model_class(dataset_path)
        
        pretrained_model_path = pretrained_models[dataset_name]['path']
        self.pretrained_vgg.load_model(pretrained_model_path)
        self.device = self.pretrained_vgg.device
        self.num_classes = self.pretrained_vgg.num_classes

        self.train_loader = self.pretrained_vgg.train_loader
        self.test_loader = self.pretrained_vgg.test_loader

        # Add noisy testdata variant
        self.noise_test_dataset = copy.deepcopy(self.pretrained_vgg.test_dataset)
        self.noise_test_dataset.transform = self._noise_transform()
        self.noise_test_loader = torch.utils.data.DataLoader(
            self.noise_test_dataset, batch_size=self.test_loader.batch_size,
            shuffle=False, num_workers=4, pin_memory=True)

        self._build_lcl_vgg()

    def _build_lcl_vgg(self):
        self.features = nn.Sequential(
            OrderedDict([
                ('pool1', self.pretrained_vgg.net.features[:5]),
                ('lcl1',  LaterallyConnectedLayer(num_crosstalk_replications, 64, 112, 112)),
                ('pool2', self.pretrained_vgg.net.features[5:10]),
                ('lcl2',  LaterallyConnectedLayer(num_crosstalk_replications, 128, 56, 56)),
                ('pool3', self.pretrained_vgg.net.features[10:19]),
                ('lcl3',  LaterallyConnectedLayer(num_crosstalk_replications, 256, 28, 28)),
                ('pool4', self.pretrained_vgg.net.features[19:28]),
                ('lcl4',  LaterallyConnectedLayer(num_crosstalk_replications, 512, 14, 14)),
                ('pool5', self.pretrained_vgg.net.features[28:]),
            ])
        )
        self.avgpool = self.pretrained_vgg.net.avgpool
        self.classifier = self.pretrained_vgg.net.classifier
        self.eval()

    def _noise_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5)),
            AddGaussianNoise(0., 1.)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def test(self, noise=False, loader=None):
        self.eval()

        if loader is None:
            if noise:
                loader = self.noise_test_loader
            else:
                loader = self.test_loader

        with torch.no_grad():
            correct = 0
            total = 0
            counter = 0

            desc = f"Evaluating {self.dataset_name} test data {'with' if noise else 'without'} noise."
            for i, (images, labels) in tqdm(enumerate(loader, 0), total=len(loader), desc=desc):
                counter += 1
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
            acc = correct/total

        return acc


class LateralModel:
    def __init__(self, vgg_model, dataset,
        distance=2, alpha=0.1, beta=0.1, gamma=0.1, eta=0.1, theta=0.1, mu_batch_size=128,
        num_output_repetitions=2, num_padding_artifact_pixels=2, num_cell_dynamic_iterations=1):
       
        self.vgg_model = vgg_model
        self.gpu_device = self.vgg_model.device
        self.dataset = dataset

        self.n = num_output_repetitions         # number of feature cell copies to reduce cross talk
        self.d = distance                       # distance
        self.k = 2 * self.d + 1                 # kernel size
        self.prd = num_padding_artifact_pixels  # Padding Repair Distance
        self.eta = eta                          # Parameter to control the amount of noise added to the feature maps
        self.mu_batch_size = mu_batch_size      # Number of images from dataset to calculate initial :mu

        self.beta = beta                        # Adjustment rate of mean output per FM
        self.gamma = gamma                      # Adjustment rate of FM strength

        self.theta = theta
        self.t_dyn = num_cell_dynamic_iterations

        if alpha >= 1 or alpha < 0:
            raise ValueError(':alpha should be between 1 and 0 (not inclusive).')
        else:
            self.alpha = alpha                  # alpha (kernel learning rate)
        
        self.iterations_trained = 0

        self.K = None
        self.K_change = None
        self.S = None
        self.M = None
        self.mu = None
        self.L = None
        self.O = None
        
    # img.shape =              (1,   3, 224, 224)
    # A.shape [INITIAL] =         (128,  56,  56)
    # A.shape [CROSSTALK] =    (n, 128,  56,  56)
    #
    def forward(self, img, update=True):
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.reshape((1,) + img.shape).float()

            img_gpu = img.to(self.gpu_device)
            
            # Run image through preprocessing network and retrieve activations
            #
            A = self.vgg_model.net(img_gpu)[0, ...]
            num_fm, num_fm_height, num_fm_width = A.shape

            # ====================================================================

            # Initialization of K, at zero except in the middle kernel pixel
            #
            if self.K is None:
                self.K = torch.rand(size=(self.n * num_fm, self.n * num_fm, self.k, self.k), device=self.gpu_device)

                # lehl@2022-02-10: Applying softmax helps to keep these values somewhat bounded and to
                # avoid over- and underflow issues by rounding to 0 or infinity for small/large exponentials
                #
                self.K = softmax_on_fm(self.K)

            # Initialization of S, M and mu, saving the scaling factor of each
            # lateral connection and its historical average
            #
            if self.S is None:
                # Generate a large batch of random images from the dataset
                #
                batch_imgs, _ = self.dataset.get_batch(self.mu_batch_size)
                A_batch = self.vgg_model.net(batch_imgs.to(self.gpu_device))

                # Calculate the mean activation of each feature map for this batch,
                # divided by the number of alternative repetitions, as the overall mean
                # should be approached by all identical feature cells together
                #
                self.mu = torch.mean(torch.sum(A_batch, dim=(-2,-1)) / (A_batch.shape[-2] * A_batch.shape[-1]), dim=0) / self.mu_batch_size
                self.mu = torch.broadcast_to(self.mu, (self.n,) + self.mu.shape).reshape((self.n * self.mu.shape[0])) / self.n

                self.S = torch.clone(self.mu)
                self.M = torch.clone(self.mu)

                del A_batch
                del batch_imgs

            # ====================================================================

            # lehl@2022-01-26: Repairing of the padding border effect by
            # applying a symmetric padding of the affected area
            #
            A[...,:self.prd,:] = torch.flip(A[...,self.prd:2*self.prd,:], dims=(-2,))         # top edge
            A[...,-self.prd:,:] = torch.flip(A[...,-2*self.prd:-self.prd,:], dims=(-2,))      # bottom edge
            A[...,:,:self.prd] = torch.flip(A[...,:,self.prd:2*self.prd], dims=(-1,))         # left edge
            A[...,:,-self.prd:] = torch.flip(A[...,:,-2*self.prd:-self.prd], dims=(-1,))      # right edge

            # Introduce N identical feature cells to reduce cross talk
            # Multiplied feature maps are treated as feature maps on the same level, i.e.
            # for activations A of shape (100, 50, 50) and n=4, we expect the new A to
            # be of shape (4*100, 50, 50) with (a, i, j) == (a + 100, i, j)
            #
            self.A = torch.broadcast_to(A, (self.n,) + A.shape).reshape((self.n * A.shape[0],) + A.shape[1:])
                
            if update:
                # lehl@2022-02-04: Add noise to help break the symmetry
                #
                noise = torch.normal(torch.zeros(self.A.shape), torch.ones(self.A.shape)).to(self.gpu_device)
                self.A = torch.clip(self.A + self.eta * noise, min=0.0)

                # Lateral Connection Reinforcement
                #
                self.K_change = torch.zeros(size=self.K.shape, device=self.gpu_device)

                for a in range(self.K.shape[1]):
                    for x in range(self.K.shape[2]):
                        for y in range(self.K.shape[3]):
                            # All the feature maps that influence the activation
                            #
                            xoff_f = - self.d + x
                            yoff_f = - self.d + y
                            influencing_feature_maps = self.A[:, max(0, xoff_f):self.A.shape[-2]+min(xoff_f, 0), max(0, yoff_f):self.A.shape[-1]+min(0, yoff_f)]
                            
                            # The activation to be influenced
                            #
                            xoff_i = self.d - x
                            yoff_i = self.d - y
                            influenced_feature_map = self.A[a, max(0, xoff_i):self.A.shape[-2]+min(xoff_i, 0), max(0, yoff_i):self.A.shape[-1]+min(0, yoff_i)]

                            # Divide by the number of feature maps used, to scale down the kernel
                            #
                            self.K_change[:, a, x, y] = torch.sum(influencing_feature_maps * influenced_feature_map, dim=(1,2)) / self.K.shape[0]
                
                # Additional normalization because of the potential number of height and width pixels summed
                #
                self.K_change /= (self.K.shape[-2] * self.K.shape[-1])

                # lehl@2022-02-10: Apply the softmax to the K changes per feature map, such
                # that we have no over- or underflows over time. 
                #
                # should hold: | alpha * K_change + (1 - alpha) * K | === 1
                #
                self.K_change = softmax_on_fm(self.K_change)
                self.K = ((1 - self.alpha) * self.K) + self.alpha * self.K_change

            # Replicate for timekeeping / debugging
            self.A = torch.broadcast_to(self.A, (self.t_dyn,) + self.A.shape)
            self.O = torch.zeros(size=self.A.shape, device=self.gpu_device)

            for t in range(self.t_dyn):
                # Applying the Kernel Convolution
                #
                # lehl@2022-01-26: When using padding=d, the resulting activation maps include
                # artefacts at the image borders where activations are unusually high/low.
                #
                # lehl@2022-01-28: Add symmetrical padding to A bevor applying the convolution,
                # then do the convolution with valid padding
                #
                self.padded_A = torch.zeros(self.A.shape[1:-2] + (self.prd*self.d + self.A.shape[-2], self.prd*self.d + self.A.shape[-1]), dtype=self.A.dtype, device=self.gpu_device)

                # copy over A into padded A
                self.padded_A[..., self.prd:-self.prd, self.prd:-self.prd] = self.A[t, ...]

                # fix borders for padded A
                # (use 2x prd, in case that VGG was already symmetrically padded -> otherwise we have two identical mirrorings)
                pad_d = 2 * self.prd

                self.padded_A[...,:pad_d,:] = torch.flip(self.padded_A[...,pad_d:2*pad_d,:], dims=(-2,))         # top edge
                self.padded_A[...,-pad_d:,:] = torch.flip(self.padded_A[...,-2*pad_d:-pad_d,:], dims=(-2,))      # bottom edge
                self.padded_A[...,:,:pad_d] = torch.flip(self.padded_A[...,:,pad_d:2*pad_d], dims=(-1,))         # left edge
                self.padded_A[...,:,-pad_d:] = torch.flip(self.padded_A[...,:,-2*pad_d:-pad_d], dims=(-1,))      # right edge
                
                # lehl@2022-02-10: Add softmax to the kernel on each feature map before the convolution step,
                # such that the unbounded ratios are bounded
                #
                self.L = F.conv2d(self.padded_A.unsqueeze(0), minmax_on_fm(self.K.transpose_(0, 1)), padding=0).squeeze() / num_fm

                # Apply Softmax to L, then MinMax Scaling, so that the final values
                # are in the range of [0, 1], reaching from 0 to 1
                #
                self.L = softmax_minmax_scaled(self.L)

                # Activation Scaling
                #
                self.O[t, ...] = self.A[t, ...] * self.L * self.S.unsqueeze(-1).unsqueeze(-1)
                
                if t < self.t_dyn - 1:
                    self.A[t+1, ...] = (1 - self.theta) * self.A[t, ...] + self.theta * self.A[t, ...] * self.L * self.S.unsqueeze(-1).unsqueeze(-1)

            if update:
                # Update the mean and down-/upscale the feature map strengths
                self.M = (1 - self.beta) * self.M + self.beta * torch.mean(self.A[-1, ...] * self.L * self.S.unsqueeze(-1).unsqueeze(-1), dim=(-2,-1))
                self.S = torch.clip(self.S + self.gamma * (self.mu - self.M), min=0.0, max=1.0)

            # Keep track of number of training iterations
            #
            self.iterations_trained += 1 if update else 0

            return self.O[-1, ...]

    def save_model(self, file_path:str=None):
        if not file_path:
            file_path = 'models/lateral_models/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.h5'

        with h5py.File(file_path, 'w') as f:
            lg = f.create_group('lateral_model')
            lg.create_dataset('K', data=self.K.cpu().detach().numpy())
            lg.create_dataset('iterations_trained', data=[self.iterations_trained])
            lg.create_dataset('d', data=[self.d])
            lg.create_dataset('k', data=[self.k])
            lg.create_dataset('alpha', data=[self.alpha])
            lg.create_dataset('prd', data=[self.prd])
            lg.create_dataset('eta', data=[self.eta])

        print(f"Saved model to {file_path} successfully.")

    def load_model(self, file_path:str):
        if not file_path:
            raise ValueError(":file_path needs to be given in order to load the model.")

        with h5py.File(file_path, 'r') as f:
            self.K = torch.Tensor(f['lateral_model']['K']).to(self.gpu_device)
            self.iterations_trained = f['lateral_model']['iterations_trained'][0]
            self.d = f['lateral_model']['d'][0]
            self.k = f['lateral_model']['k'][0]
            self.alpha = f['lateral_model']['alpha'][0]
            self.prd = f['lateral_model']['prd'][0]
            self.eta = f['lateral_model']['eta'][0]

        print(f"Loaded model from {file_path} successfully.\t({self.iterations_trained} train iterations detected.)")
        print(f"\t\tConfig used:\td: {self.d}, k: {self.k}, alpha: {self.alpha}")


if __name__ == '__main__':
    m = VggLateral('MNIST')

    import code; code.interact(local=dict(globals(), **locals()))
