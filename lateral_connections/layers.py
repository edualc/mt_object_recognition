import torch
import torch.nn.functional as F
import torch.nn as nn

from .torch_utils import *


# PyTorch layer implementation of the laterally connected layer (LCL)
#
# Arguments:
# - n:              Number of multiplex cells (repetitions to reduce crosstalk)
# - num_fm:         Number of input feature maps
# - fm_height:      Height of input feature maps (in pixels)
# - fm_width:       Width of input feature maps (in pixels)
# - d:              Distance away from center pixel in the kernel, for 3x3 choose 1
# - prd:            Padding repair distance, how thick the border around activations is
#                       fixed with symmetric padding, to alleviate border effect artifacts
#
# TODO: Check out torchfund for performance tips / hooks
#   https://github.com/szymonmaszke/torchfunc
# Pytorch Hooks:
#   https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
#
class LaterallyConnectedLayer(nn.Module):
    def __init__(self, n, num_fm, fm_height, fm_width, d=2, prd=2, disabled=True, update=True,
        alpha=0.1, beta=0.01, gamma=0.01, eta=0.1, theta=0.1, mu_batch_size=128):
        super().__init__()
        self.disabled = disabled
        
        # Use the pyTorch internal self.training, which gets adjusted
        # by model.eval() / model.train(), to control whether updates should be done
        self.training = update
        self.iterations_trained = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n = n
        self.d = d
        self.k = 2*self.d + 1
        self.prd = prd
        self.num_fm = num_fm
        self.fm_height = fm_height
        self.fm_width = fm_width

        self.alpha = alpha                          # Rate at which the kernel K is changed by K_change
        self.beta = beta                            # Adjustment rate of mean output per FM
        self.gamma = gamma                          # Adjustment rate of FM strength
        self.eta = eta                              # Parameter to control the amount of noise added to the feature maps
        self.theta = theta                          # Rate at which each cell dynamic iteration changes A
        self.mu_batch_size = mu_batch_size          # Number of images from dataset to calculate initial :mu

        # lehl@2022-02-10: Applying softmax helps to keep these values somewhat bounded and to
        # avoid over- and underflow issues by rounding to 0 or infinity for small/large exponentials
        K = softmax_on_fm(torch.rand(size=(self.n * self.num_fm, self.n * self.num_fm, self.k, self.k)))
        self.register_parameter('K', torch.nn.Parameter(K, requires_grad=False))
        del K

        self.L = None
        self.O = None

        # Initialization of S, M and mu, saving the scaling factor of each
        # lateral connection and its historical average
        # TODO: Initialize with a sensible mean, taken from a sample of activations
        self.register_parameter('mu', torch.nn.Parameter(torch.ones((self.n * self.num_fm,)) / self.n, requires_grad=False))
        self.register_parameter('S', torch.nn.Parameter(torch.clone(self.mu), requires_grad=False))
        self.register_parameter('M', torch.nn.Parameter(torch.clone(self.mu), requires_grad=False))

    def __repr__(self):
        return f"{'' if self.disabled else '*'}{self.__class__.__name__}({self.n}, ({self.num_fm}, {self.fm_height}, {self.fm_width}), d={str(self.d)}, disabled={str(self.disabled)}, update={str(self.training)})"

    def pad_activations(self, A):
        padded_A = torch.zeros(A.shape[:-2] + (self.prd*self.d+A.shape[-2], self.prd*self.d+A.shape[-1]), dtype=A.dtype, device=self.device)
        padded_A[..., self.prd:-self.prd, self.prd:-self.prd] = A
        return symmetric_padding(padded_A, 2*self.prd)

    # Keep the layer at this stage and no longer update relevant matrices,
    # only allow the layer to perform inference
    #
    def freeze(self):
        self.training = False

    # Allow updates to happen on the layer
    #
    def unfreeze(self):
        self.training = True

    def enable(self):
        self.disabled = False

    # Disable the layer, turning it into an identity function that passes
    # the given input as its output without any transformations
    #
    def disable(self):
        self.disabled = True
    
    # This method expects a batch of m images to be passed as A,
    # giving it the shape (m, F, H, W)
    #
    def forward(self, A):
        if self.disabled:
            return torch.clone(A)

        # # TODO: Pavel said no
        # # check if grad is still there in layers around
        # with torch.no_grad():
        # import code; code.interact(local=dict(globals(), **locals()))

        with torch.no_grad():
            batch_size = A.shape[0]

            # Symmetric padding fix to remove border effect issues from 0-padding
            symm_A = symmetric_padding(A, self.prd)

            # Generate multiplex cells
            # ---
            # Introduce N identical feature cells to reduce cross talk
            # Multiplied feature maps are treated as feature maps on the same level, i.e.
            # for activations A of shape (32, 100, 50, 50) and n=4, we expect the new A to
            # be of shape (32, 4*100, 50, 50) with (bs, a, i, j) == (bs, a + 100, i, j)
            #
            self.A = symm_A.repeat(1, self.n, 1, 1)

            # Add noise to help break the symmetry between initially
            # identical multiplex cells
            #
            noise = torch.normal(torch.zeros(self.A.shape), torch.ones(self.A.shape)).to(self.device)
            self.A = torch.clip(self.A + self.eta * noise, min=0.0)

            # generate padded A for convolution
            padded_A = self.pad_activations(self.A)
            
            # Calculate initial lateral connections
            self.L = F.conv2d(padded_A, minmax_on_fm(self.K.transpose_(0, 1)), padding=0) / self.num_fm
            self.L = softmax_minmax_scaled(self.L)
            del padded_A

            # Calculate A_hat, perform convolution (L) and add to intial input
            # find the maximum multiplex cells for each multiple (n)
            A_max = torch.sum(self.A + self.L, dim=(-2,-1))
            A_max = A_max.reshape((A_max.shape[0], self.n, A_max.shape[1]//self.n))
            A_max = torch.argmax(torch.transpose(A_max,2,1), dim=-1)

            # Get the argmax indices inside the multiplex cells for each sample in the batch
            fm_indices = A_max.shape[1] * A_max + torch.arange(0, A_max.shape[1]).to(self.device)
            filtered_A = torch.stack([torch.index_select(i, dim=0, index=j) for i,j in zip(self.A, fm_indices)])
            del A_max

            k_indices = torch.repeat_interleave(fm_indices.unsqueeze(-1), fm_indices.shape[1], dim=2)

            # Build idx of disabled multiplex cells
            inactive_multiplex_idx = torch.zeros((batch_size, self.A.shape[1] - fm_indices.shape[1]), device=self.device).to(torch.long)
            for b in range(batch_size):
                uniqs, cnts = torch.cat((torch.arange(self.A.shape[1]).to(self.device), fm_indices[b,:])).unique(return_counts=True)
                inactive_multiplex_idx[b,:] = uniqs[cnts==1]
            del uniqs
            del cnts

            if self.training:
                # Lateral Connection Reinforcement
                K_change = torch.zeros(size=self.K.shape, device=self.device)

                # Iterate through the different necessary shifts between the feature maps
                # to account for all positions in the kernel K
                #
                for x in range(self.K.shape[-2]):
                    for y in range(self.K.shape[-1]):
                        # source and target feature maps are transposed to enable broadcasting across
                        # the batch size dimension
                        #
                        xoff_f = - self.d + x
                        yoff_f = - self.d + y
                        source_fms = self.A[:, :, max(0,xoff_f):self.A.shape[-2]+min(xoff_f,0), max(0,yoff_f):self.A.shape[-1]+min(0,yoff_f)]
                        source_fms = source_fms.transpose(0,1)

                        xoff_i = self.d - x
                        yoff_i = self.d - y
                        target_fms = self.A[:, :, max(0,xoff_i):self.A.shape[-2]+min(xoff_i,0), max(0,yoff_i):self.A.shape[-1]+min(0,yoff_i)]
                        target_fms = target_fms.transpose(0,1).unsqueeze(1)

                        # calculate the product of all feature maps (source) together with the
                        # to-be-influenced feature map (target) efficiently
                        #
                        tmp = torch.einsum('abcde,bcde->cab', target_fms, source_fms)
                        
                        # inhibit inactive multiplex cell changes
                        #
                        for b in range(batch_size):
                            tmp[b, inactive_multiplex_idx[b,:]] = 0

                        # Average across the batch size
                        #
                        K_change[:, :, x, y] = torch.sum(tmp, dim=0) / batch_size
                        # import code; code.interact(local=dict(globals(), **locals()))

                K_change = minmax_on_fm(K_change)
                
                # lehl@2022-02-10: Apply the softmax to the K changes per feature map, such
                # that we have no over- or underflows over time. 
                #
                # should hold: | alpha * K_change + (1 - alpha) * K | === 1
                #
                self.K *= (1 - self.alpha)
                self.K += self.alpha * K_change
                del K_change

        # Calculate output
        tmp_A = torch.clone(self.A)
        large_K = self.K.repeat(batch_size, 1, 1, 1, 1)
        for b in range(batch_size):
            large_K[b, inactive_multiplex_idx[b,:], inactive_multiplex_idx[b,:], :, :] = 0
            tmp_A[b, inactive_multiplex_idx[b,:], ...] = 0

        # Applying the Kernel Convolution
        #
        # lehl@2022-01-26: When using padding=d, the resulting activation maps include
        # artefacts at the image borders where activations are unusually high/low.
        #
        # lehl@2022-01-28: Add symmetrical padding to A bevor applying the convolution,
        # then do the convolution with valid padding
        #
        padded_A = self.pad_activations(tmp_A)

        L = torch.zeros((batch_size, self.n*self.num_fm, self.fm_height, self.fm_width), device=self.device)

        for b in range(batch_size):
            L[b, ...] = F.conv2d(padded_A[b, ...].unsqueeze(0), minmax_on_fm(large_K[b,...].transpose(0,1)), padding=0) / self.num_fm

        filtered_L = torch.stack([torch.index_select(i, dim=0, index=j) for i,j in zip(L, fm_indices)])
        self.O = A + self.theta * filtered_L

        # lehl@2022-03-30: TODO - Is this still needed?
        # if self.training:
        #     # Update the mean and down-/upscale the feature map strengths
        #     self.M = (1 - self.beta) * self.M + self.beta * torch.mean(self.A[-1, ...] * self.L * self.S.unsqueeze(-1).unsqueeze(-1), dim=(-2,-1))
        #     self.S = torch.clip(self.S + self.gamma * (self.mu - self.M), min=0.0, max=1.0)

        # Keep track of number of training iterations
        self.iterations_trained += 1 if self.training else 0

        return self.O
