import numpy as np
import torch
import torch.nn.functional as F

from .vgg_model import VggModel
from .dataset import CustomImageDataset
from .distance_map import DistanceMapTemplate, ManhattanDistanceTemplate

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import h5py


class LateralModel:
    def __init__(self, vgg_model, num_output_repetitions=2,
        horizon_length=200, expected_duty_cycle=0.005, distance=1, alpha=0.1, delta=0.1, num_padding_artifact_pixels=2):
       
        self.vgg_model = vgg_model
        self.gpu_device = self.vgg_model.device

        self.n = num_output_repetitions         # number of feature cell copies to reduce cross talk
        self.d = distance                       # distance
        self.h = horizon_length                 # horizon length
        self.k = 2 * self.d + 1                 # kernel size
        self.mu = expected_duty_cycle           # expected duty cycle
        self.delta = delta                      # saturation increase or decrease
        self.prd = num_padding_artifact_pixels  # Padding Repair Distance

        if alpha >= 1 or alpha < 0:
            raise ValueError(':alpha should be between 1 and 0 (not inclusive).')
        else:
            self.alpha = alpha                  # alpha (kernel learning rate)
        
        self.iterations_trained = 0

        self.O = None
        self.K = None
        self.H = None
        self.S = None
        self.L = None
        
    def save_model(self, file_path:str=None):
        if not file_path:
            file_path = 'models/lateral_models/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.h5'

        with h5py.File(file_path, 'w') as f:
            lg = f.create_group('lateral_model')
            lg.create_dataset('K', data=self.K.cpu().detach().numpy())
            lg.create_dataset('H', data=self.H.cpu().detach().numpy())
            lg.create_dataset('iterations_trained', data=[self.iterations_trained])
            lg.create_dataset('d', data=[self.d])
            lg.create_dataset('k', data=[self.k])
            lg.create_dataset('h', data=[self.h])
            lg.create_dataset('mu', data=[self.mu])
            lg.create_dataset('alpha', data=[self.alpha])

        print(f"Saved model to {file_path} successfully.")

    def load_model(self, file_path:str):
        if not file_path:
            raise ValueError(":file_path needs to be given in order to load the model.")

        with h5py.File(file_path, 'r') as f:
            self.K = torch.Tensor(f['lateral_model']['K']).to(self.gpu_device)
            self.H = torch.Tensor(f['lateral_model']['H']).to(self.gpu_device)
            self.iterations_trained = f['lateral_model']['iterations_trained'][0]
            self.d = f['lateral_model']['d'][0]
            self.k = f['lateral_model']['k'][0]
            self.h = f['lateral_model']['h'][0]
            self.mu = f['lateral_model']['mu'][0]
            self.alpha = f['lateral_model']['alpha'][0]

        print(f"Loaded model from {file_path} successfully.\t({self.iterations_trained} train iterations detected.)")
        print(f"\t\tConfig used:\td: {self.d}, k: {self.k}, h: {self.h}, mu: {self.mu}, alpha: {self.alpha}")

    # img.shape =              (1,   3, 224, 224)
    # A.shape [INITIAL] =         (128,  56,  56)
    # A.shape [CROSSTALK] =    (n, 128,  56,  56)
    #
    def forward(self, img, update=True, do_sympad_vgg=True, do_sympad_lateral=True):
        with torch.no_grad():
            img_gpu = img.to(self.gpu_device)
            
            A = self.vgg_model.net(img_gpu)[0, ...]

            # Threshold Clipping, as the output of VGG19's pooling layers
            # use ReLU and are not guaranteed to stay within [0, 1]
            #
            A = torch.clip(A, min=0.0, max=1.0)

            num_fm, num_fm_height, num_fm_width = A.shape
            
            # ====================================================================

            # Initialization of K, at zero except in the middle kernel pixel
            #
            if self.K is None:
                self.K = torch.zeros(size=(self.n * num_fm, self.n * num_fm, self.k, self.k)).to(self.gpu_device)
                self.K[:, self.d, self.d] = 1
            
            if self.H is None:
                self.H = torch.zeros(size=(self.n * A.shape[0],) + A.shape[1:]).to(self.gpu_device)
                
            if self.S is None:
                self.S = torch.ones(size=(self.n * A.shape[0],) + A.shape[1:]).to(self.gpu_device)
            
            # ====================================================================
                

            # lehl@2022-01-26: Repairing of the padding border effect by
            # applying a symmetric padding of the affected area
            #
            if do_sympad_vgg:
                A[...,:self.prd,:] = torch.flip(A[...,self.prd:2*self.prd,:], dims=(-2,))         # top edge
                A[...,-self.prd:,:] = torch.flip(A[...,-2*self.prd:-self.prd,:], dims=(-2,))      # bottom edge
                A[...,:,:self.prd] = torch.flip(A[...,:,self.prd:2*self.prd], dims=(-1,))         # left edge
                A[...,:,-self.prd:] = torch.flip(A[...,:,-2*self.prd:-self.prd], dims=(-1,))      # right edge

            # Introduce N identical feature cells to reduce cross talk
            # Multiplied feature maps are treated as feature maps on the same level, i.e.
            # for activations A of shape (100, 50, 50) and n=4, we expect the new A to
            # be of shape (4*100, 50, 50) with (a, i, j) == (a + 100, i, j)
            #
            self.A = torch.broadcast_to(A, (self.n,) + A.shape).reshape((self.n * A.shape[0],) + A.shape[1:]).to(self.gpu_device)

            if update:
                # Lateral Connection Reinforcement
                #
                self.K *= (1 - self.alpha)

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

                            self.K[:, a, x, y] += self.alpha * torch.sum(influencing_feature_maps * influenced_feature_map, dim=(1,2))

                            # if a == 0:
                            #     print(f"a:{a}, x:{x}, y:{y}\tA:{self.A.shape}, A2-x:{self.A.shape[-2]-x}, A1-y:{self.A.shape[-1]-y}")

                            # self.K[:, a, x, y] += self.alpha * torch.sum(self.A[:, x:, y:] * self.A[a, :self.A.shape[-2]-x, :self.A.shape[-1]-y], dim=(1,2))
                            # # self.K[:, a, x, y] += self.alpha * torch.sum(self.A[:, x:, y:] * self.A[a, :self.A.shape[-2]-x, :self.A.shape[-1]-y], dim=(1,2)) / (self.k ** 2)
            
                # Kernel Normalization
                #
                # lehl@2022-01-27: It looks like some kernels can sum to zero, which would introduce
                # NaN values in the normalization division. Instead, divide by 1 in those cases.
                #
                divisor = torch.sum(self.K, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
                divisor[divisor==0] = 1.0
                self.K /= divisor

            # Applying the Kernal Convolution
            #
            # lehl@2022-01-26: When using padding=d, the resulting activation maps include
            # artefacts at the image borders where activations are unusually high/low.
            #
            # lehl@2022-01-28: Add symmetrical padding to A bevor applying the convolution,
            # then do the convolution with valid padding
            #
            if do_sympad_lateral:
                self.padded_A = torch.zeros(self.A.shape[:-2] + (self.prd*self.d + self.A.shape[-2], self.prd*self.d + self.A.shape[-1]), dtype=self.A.dtype).to(self.gpu_device)

                # copy over A into padded A
                self.padded_A[..., self.prd:-self.prd, self.prd:-self.prd] = self.A

                # fix borders for padded A
                # (use 2x prd, in case that VGG was already symmetrically padded -> otherwise we have two identical mirrorings)
                pad_d = self.prd
                if do_sympad_vgg:
                    pad_d *= 2

                self.padded_A[...,:pad_d,:] = torch.flip(self.padded_A[...,pad_d:2*pad_d,:], dims=(-2,))         # top edge
                self.padded_A[...,-pad_d:,:] = torch.flip(self.padded_A[...,-2*pad_d:-pad_d,:], dims=(-2,))      # bottom edge
                self.padded_A[...,:,:pad_d] = torch.flip(self.padded_A[...,:,pad_d:2*pad_d], dims=(-1,))         # left edge
                self.padded_A[...,:,-pad_d:] = torch.flip(self.padded_A[...,:,-2*pad_d:-pad_d], dims=(-1,))      # right edge

                self.L = torch.clip(F.conv2d(self.padded_A.unsqueeze(0), self.K.transpose_(0, 1), padding=0).squeeze() / num_fm, min=0.0, max=1.0)

            else:
                self.L = torch.clip(F.conv2d(self.A.unsqueeze(0), self.K.transpose_(0, 1), padding=self.d).squeeze() / num_fm, min=0.0, max=1.0)

            # Apply Softmax to L, then MinMax Scaling, so that the final values
            # are in the range of [0, 1], reaching from 0 to 1
            #
            #self.L = torch.nn.Softmax2d()(self.L.unsqueeze(0)).squeeze()

            # self.L = F.conv2d(self.A.unsqueeze_(0), self.K.transpose_(0, 1), padding=self.d).squeeze() / (num_fm * (self.k ** 2))
            # self.L = torch.clip(F.conv2d(self.A.unsqueeze_(0), self.K.transpose_(0, 1), padding=self.d).squeeze() / (num_fm * (self.k ** 2)), min=0.0, max=1.0)

            if update:
                # Update of Cell Activity History
                #
                h_factor = torch.divide(torch.Tensor([1]), torch.Tensor([self.h])).to(self.gpu_device)

                self.H = torch.clip(h_factor * self.L + (1 - h_factor) * self.H, min=0.0, max=1.0)

                # Cell Activity Normalization
                #
                # self.S = torch.ones(size=(self.n * A.shape[0],) + A.shape[1:]).to(self.gpu_device)
                # self.S *= torch.ones(size=self.S.shape).to(self.gpu_device) + torch.mul(torch.sign(self.mu - self.H), self.delta)
                # self.S = torch.clip(self.S, min=0.5, max=2.0)

                # torch.clip(torch.ones(shape=self.S.shape) + torch.mul(torch.sign(self.mu - self.H), self.delta), min=0.9, max=)
                self.S = torch.clip(self.mu - self.H, min=-1.0, max=1.0)

                # Keep track of number of training iterations
                self.iterations_trained += 1

            # Activation Scaling
            #
            self.O = self.A * self.L
            # self.O = self.A * self.L * self.S

            return self.O
