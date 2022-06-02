import torch.nn.functional as F
import torch.nn as nn

from .torch_utils import *

import wandb

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
        alpha=0.1, beta=0.001, gamma=0.001, eta=1, theta=0.0, iota=0.1, mu_batch_size=0, num_noisy_iterations=1000,
        use_scaling=True, random_k_change=False, random_multiplex_selection=False, gradient_learn_k=False):
        super().__init__()
        self.disabled = disabled
        
        # Use the pyTorch internal self.training, which gets adjusted
        # by model.eval() / model.train(), to control whether updates should be done
        self.training = update
        self.iterations_trained = 0
        self.num_noisy_iterations = num_noisy_iterations

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_scaling = use_scaling

        # Ablation study variants
        self.random_k_change = random_k_change
        self.random_multiplex_selection = random_multiplex_selection
        self.gradient_learn_k = gradient_learn_k

        self.register_parameter('n', nn.Parameter(torch.Tensor([n]).to(torch.int), requires_grad=False))
        self.register_parameter('d', nn.Parameter(torch.Tensor([d]).to(torch.int), requires_grad=False))
        self.register_parameter('k', nn.Parameter(torch.Tensor([2*self.d+1]).to(torch.int), requires_grad=False))
        self.register_parameter('prd', nn.Parameter(torch.Tensor([prd]).to(torch.int), requires_grad=False))
        self.register_parameter('num_fm', nn.Parameter(torch.Tensor([num_fm]).to(torch.int), requires_grad=False))
        self.register_parameter('fm_height', nn.Parameter(torch.Tensor([fm_height]).to(torch.int), requires_grad=False))
        self.register_parameter('fm_width', nn.Parameter(torch.Tensor([fm_width]).to(torch.int), requires_grad=False))

        # Rate at which the kernel K is changed by K_change
        self.register_parameter('alpha', nn.Parameter(torch.Tensor([alpha]), requires_grad=False))
        # Adjustment rate of mean output per FM
        self.register_parameter('beta', nn.Parameter(torch.Tensor([beta]), requires_grad=False))
        # Adjustment rate of FM strength
        self.register_parameter('gamma', nn.Parameter(torch.Tensor([gamma]), requires_grad=False))
        # How the (1-iota)*A + iota*L argmax is calculated
        self.register_parameter('iota', nn.Parameter(torch.Tensor([iota]), requires_grad=False))
        

        # DEPRECATED:
        # Parameter to control the amount of noise added to the feature maps
        self.register_parameter('eta', nn.Parameter(torch.Tensor([eta]), requires_grad=False))
        # Rate at which each cell dynamic iteration changes A
        self.register_parameter('theta', nn.Parameter(torch.Tensor([theta]), requires_grad=False))
        # Number of images from dataset to calculate initial :mu
        self.register_parameter('mu_batch_size', nn.Parameter(torch.Tensor([mu_batch_size]), requires_grad=False))


        # lehl@2022-02-10: Applying softmax helps to keep these values somewhat bounded and to
        # avoid over- and underflow issues by rounding to 0 or infinity for small/large exponentials
        K = torch.zeros((self.n*self.num_fm, self.n*self.num_fm, self.k, self.k))
        #K = softmax_on_fm(torch.rand(size=(self.n * self.num_fm, self.n * self.num_fm, self.k, self.k)))
        self.register_parameter('K', torch.nn.Parameter(K, requires_grad=False))
        del K

        # To keep track of which ones are selected
        self.current_k_indices = None

        self.L = None
        self.O = None

        # Initialization of S, M and mu, saving the scaling factor of each
        # lateral connection and its historical average
        # TODO: Initialize with a sensible mean, taken from a sample of activations
        self.register_parameter('mu', torch.nn.Parameter(torch.ones((self.n * self.num_fm,)) / self.n, requires_grad=False))
        self.register_parameter('S', torch.nn.Parameter(torch.ones(self.mu.shape), requires_grad=False))
        self.register_parameter('M', torch.nn.Parameter(torch.clone(self.mu), requires_grad=False))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n.item()}, ({self.num_fm.item()}, {self.fm_height.item()}, {self.fm_width.item()}), d={str(self.d.item())}, disabled={str(self.disabled)}, update={str(self.training)})"

    def pad_activations(self, A):
        padded_A = torch.zeros(A.shape[:-2] + (self.prd+self.d+A.shape[-2], self.prd+self.d+A.shape[-1]), dtype=A.dtype, device=self.device)
        padded_A[..., self.prd:-self.prd, self.prd:-self.prd] = A
        return padded_A
        # lehl@2022-04-23: This should no longer be done, as too much of the feature
        # maps are replaced!
        #
        # return symmetric_padding(padded_A, 2*self.prd)

    # Keep the layer at this stage and no longer update relevant matrices,
    # only allow the layer to perform inference
    #
    def freeze(self):
        self.training = False # pytorch flag

    # Allow updates to happen on the layer
    #
    def unfreeze(self):
        self.training = True # pytorch flag

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

        with torch.no_grad():
            batch_size = A.shape[0]

            # Generate multiplex cells
            # ---
            # Introduce N identical feature cells to reduce cross talk
            # Multiplied feature maps are treated as feature maps on the same level, i.e.
            # for activations A of shape (32, 100, 50, 50) and n=4, we expect the new A to
            # be of shape (32, 4*100, 50, 50) with (bs, a, i, j) == (bs, a + 100, i, j)
            #
            self.A = A.repeat(1, self.n, 1, 1)

            if self.training:
                # lehl@2022-05-11: Turns out that noise makes the multiplex cells not specialize
                # enough and should --- after some initial training --- be turned off
                #
                if self.iterations_trained < self.num_noisy_iterations:
                    noise_multiplier = 1

                    # Gradually remove the noise over time
                    #
                    if self.iterations_trained > self.num_noisy_iterations // 2:
                        noise_multiplier = (self.num_noisy_iterations - self.iterations_trained) / (self.num_noisy_iterations // 2)

                    # Add noise to help break the symmetry between initially
                    # identical multiplex cells
                    #
                    noise = torch.normal(torch.zeros(self.A.shape), torch.ones(self.A.shape)).to(self.device)
                    self.A = torch.clip(self.A + self.eta * noise * noise_multiplier, min=0.0)

            # generate padded A for convolution
            padded_A = self.pad_activations(self.A)
            
            # Calculate initial lateral connections
            self.L = F.conv2d(padded_A, minmax_on_fm(self.K.transpose(0, 1)), padding=0) / self.num_fm

            # Apply the feature map scaling
            if self.use_scaling:
                self.L *= self.S.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            self.L = minmax_on_fm(self.L)
            del padded_A

            if self.random_multiplex_selection or self.iterations_trained < self.num_noisy_iterations:
                # Do the selection of multiplex cells randomly
                A_max = torch.rand((self.A.shape[0], self.n, self.A.shape[1]//self.n)).to(self.device)
                A_max = torch.argmax(torch.transpose(A_max,2,1), dim=-1)

            else:
                # Calculate A_hat, perform convolution (L) and add to intial input
                # find the maximum multiplex cells for each multiple (n)
                A_max = torch.sum((1 - self.iota) * self.A + self.iota * self.L, dim=(-2,-1))
                A_max = A_max.reshape((A_max.shape[0], self.n, A_max.shape[1]//self.n))
                A_max = torch.argmax(torch.transpose(A_max,2,1), dim=-1)

            # Get the argmax indices inside the multiplex cells for each sample in the batch
            fm_indices = A_max.shape[1] * A_max + torch.arange(0, A_max.shape[1]).to(self.device)
            # del A_max
            filtered_A = torch.stack([torch.index_select(i, dim=0, index=j) for i,j in zip(self.A, fm_indices)])

            k_indices = torch.repeat_interleave(fm_indices.unsqueeze(-1), fm_indices.shape[1], dim=2)
            self.current_k_indices = torch.clone(k_indices)

            # Build idx of disabled multiplex cells
            inactive_multiplex_idx = torch.zeros((batch_size, self.A.shape[1] - fm_indices.shape[1]), device=self.device).to(torch.long)
            for b in range(batch_size):
                uniqs, cnts = torch.cat((torch.arange(self.A.shape[1]).to(self.device), fm_indices[b,:])).unique(return_counts=True)
                inactive_multiplex_idx[b,:] = uniqs[cnts==1]
            del uniqs
            del cnts

            if self.training:
                if self.random_k_change:
                    K_change = torch.rand(size=self.K.shape, device=self.device)

                else:
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
                            # Initially, self.A contains [batch_size, num_fm, fm_height, fm_width] dimensions
                            #
                            # Designations of einsum characters:
                            #   a:  Extra dim to ensure each FM in target is multipled with the whole source blob
                            #   b:  Feature Map #
                            #   c:  Batch (see the transpose(0,1) calls)
                            #   d:  Feature Map Height
                            #   e:  Feature Map Width
                            #
                            tmp = torch.einsum('abcde,bcde->cab', target_fms, source_fms)

                            # inhibit inactive multiplex cell changes
                            #
                            for b in range(batch_size):
                                # lehl@2022-04-18: Only source and target feature maps that are active
                                # should be changed/influenced by the changes calculated here
                                #
                                tmp[b, inactive_multiplex_idx[b,:], :] = 0
                                tmp[b, :, inactive_multiplex_idx[b,:]] = 0

                            # Average across the batch size
                            #
                            number_of_samples_per_pixel = torch.count_nonzero(tmp, dim=0)
                            number_of_samples_per_pixel[torch.where(number_of_samples_per_pixel == 0)] = 1
                            K_change[:, :, x, y] = torch.sum(tmp, dim=0) / number_of_samples_per_pixel

                # K_change = minmax_on_fm(K_change)
                
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
        
        # self.O = (1 - self.theta) * A + self.theta * filtered_L
        # import code; code.interact(local=dict(globals(), **locals()))

        if self.training and self.use_scaling:
            with torch.no_grad():
                # Gather which feature maps were active in the batch and calculate the
                # mean activity
                #
                repeat_tensor = torch.arange(self.n.item()).to(self.device)
                x_idx = torch.repeat_interleave(repeat_tensor, torch.clone(self.num_fm).to(torch.long))
                y_idx = fm_indices.reshape((torch.numel(fm_indices),))

                # Calculate how active the feature maps in L were and
                # average across the batch
                L_act = torch.zeros((L.shape[0], self.num_fm * self.n)).to(torch.long).to(self.device)
                L_act.index_put_(indices=[x_idx, y_idx], values=torch.tensor(1).to(self.device))
                L_act = torch.sum(L_act, dim=0) / self.n

                print(L_act)

                # Update the mean and down-/upscale the feature map strengths
                self.M *= (1-self.beta)
                self.M += self.beta * L_act

                self.S += self.gamma * (self.mu - self.M)
                self.S = nn.Parameter(torch.clip(self.S, min=0.5, max=1.5))

        # Keep track of number of training iterations
        self.iterations_trained += 1 if self.training else 0

        return filtered_L
        # return self.O

class LaterallyConnectedLayer3(LaterallyConnectedLayer):
    def forward(self, A):
        if self.disabled:
            return torch.clone(A)

        with torch.no_grad():
            batch_size = A.shape[0]

            # Generate multiplex cells
            # ---
            # Introduce N identical feature cells to reduce cross talk
            # Multiplied feature maps are treated as feature maps on the same level, i.e.
            # for activations A of shape (32, 100, 50, 50) and n=4, we expect the new A to
            # be of shape (32, 4*100, 50, 50) with (bs, a, i, j) == (bs, a + 100, i, j)
            #
            self.A = A.repeat(1, self.n, 1, 1)

            padded_A = self.pad_activations(self.A)

            # =======================================================================================
            #   LATERAL ACTIVITY IMPACT CALCULATION
            # =======================================================================================
            #
            impact = []
            for idx in range(padded_A.shape[1]):
                kernel = self.K[:, idx, ...].unsqueeze(1)
                lateral_impact = F.conv2d(padded_A, kernel, groups=int(self.n*self.num_fm), padding=0)

                if self.use_scaling:
                    lateral_impact *= self.S.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                impact.append(torch.sum(lateral_impact.unsqueeze(2), dim=(-2,-1)))

            # impact shape = [batch, source_fm, target_fm]
            #
            impact = torch.cat(impact, dim=2)

            # =======================================================================================
            #   NOISE ADDITION
            # =======================================================================================
            #
            if self.training:
                # lehl@2022-05-11: Turns out that noise makes the multiplex cells not specialize
                # enough and should --- after some initial training --- be turned off
                #
                if self.iterations_trained < self.num_noisy_iterations:
                    noise_multiplier = 1

                    # Gradually remove the noise over time
                    #
                    if self.iterations_trained > self.num_noisy_iterations // 2:
                        noise_multiplier = (self.num_noisy_iterations - self.iterations_trained) / (self.num_noisy_iterations // 2)

                    # Add noise to help break the symmetry between initially
                    # identical multiplex cells
                    #
                    noise = noise_multiplier * torch.normal(torch.zeros(impact.shape), torch.ones(impact.shape)).to(self.device)
                    impact += noise

            # =======================================================================================
            #   ACTIVE MULTIPLEX SELECTION
            # =======================================================================================
            #
            # Remove self-references from lateral impact (wrt. multiplex repetitions)
            #
            diagonal_repetition_mask = 1 - torch.eye(self.num_fm.item(), device=self.device).repeat(self.n, self.n)
            diagonal_repetition_mask += torch.eye(int(self.num_fm*self.n), device=self.device)
            impact *= diagonal_repetition_mask.unsqueeze(0)

            # Select the highest active multiplex cell and disable others
            #
            # Implementation uses the answer provided here:
            # https://discuss.pytorch.org/t/set-max-value-to-1-others-to-0/44350/8
            # (calculate argmax along the different multiplex cell repetitions)
            #
            impact_reshaped = impact.reshape((batch_size, self.n, self.num_fm, self.n * self.num_fm))

            idx = torch.argmax(impact_reshaped, dim=1, keepdims=True)
            active_multiplex_mask = torch.zeros_like(impact_reshaped).scatter_(1, idx, 1.)
            active_multiplex_mask = active_multiplex_mask.reshape((batch_size, self.n*self.num_fm, self.n*self.num_fm))

            impact *= active_multiplex_mask

            # Taking a quantile threshold for each target_fm (= across source_fms)
            #
            # (!!!)     lehl@2022-06-02:
            #           TODO2 - Change from 20% selection at source to 20% at target (source: dim=1, target: dim=2)
            #
            impact_threshold = torch.quantile(impact.reshape((impact.shape[0], impact.shape[1]*impact.shape[2])), 0.8, dim=1)
            indices = torch.where(impact >= impact_threshold.unsqueeze(-1).unsqueeze(-1))
            # impact_threshold = torch.quantile(impact, 0.8, dim=1)
            # indices = torch.where(impact >= impact_threshold.unsqueeze(1))
            # impact_threshold = torch.quantile(impact, 0.8, dim=2)
            # indices = torch.where(impact >= impact_threshold.unsqueeze(2))

            large_K = self.K.repeat(batch_size, 1, 1, 1, 1)
            selected_multiplex_mask = torch.zeros(large_K.shape, device=self.device, dtype=torch.bool)
            selected_multiplex_mask[indices] = 1.0

            large_K *= selected_multiplex_mask

            if not wandb.run.disabled:
                if ((self.iterations_trained % 50) == 0) and self.training:
                    wandb.log({
                        'lcl': {
                            'impact': {
                                'min': round(impact.min().item(), 4),
                                'max': round(impact.max().item(), 4),
                                'mean': round(impact.mean().item(), 4)
                            },
                            'num_active_mlpx': torch.sum(active_multiplex_mask),
                            'num_selected_mlpx': torch.sum(selected_multiplex_mask),
                            'multiplex_selection': torch.sum(selected_multiplex_mask[:,:,:,0,0], dim=(0,2))/batch_size
                        }
                    })

            # =======================================================================================
            #   KERNEL UPDATE
            # =======================================================================================
            #
            if self.training:
                if self.random_k_change:
                    K_change = torch.rand(size=self.K.shape, device=self.device)

                else:
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
                            # Initially, self.A contains [batch_size, num_fm, fm_height, fm_width] dimensions
                            #
                            # Designations of einsum characters:
                            #   a:  Extra dim to ensure each FM in target is multipled with the whole source blob
                            #   b:  Feature Map #
                            #   c:  Batch (see the transpose(0,1) calls)
                            #   d:  Feature Map Height
                            #   e:  Feature Map Width
                            #
                            tmp = torch.einsum('abcde,bcde->cab', target_fms, source_fms)

                            # inhibit inactive multiplex cell changes
                            #
                            tmp *= selected_multiplex_mask[:,:,:,0,0]

                            # inhibit multiplex repetitions
                            #
                            diagonal_repetition_mask = 1 - torch.eye(self.num_fm.item(), device=self.device).repeat(self.n, self.n)
                            diagonal_repetition_mask += torch.eye(int(self.num_fm*self.n), device=self.device)
                            tmp *= diagonal_repetition_mask.unsqueeze(0)

                            # Average across the batch size
                            #
                            number_of_samples_per_pixel = torch.count_nonzero(tmp, dim=0)
                            number_of_samples_per_pixel[torch.where(number_of_samples_per_pixel == 0)] = 1
                            K_change[:, :, x, y] = torch.sum(tmp, dim=0) / number_of_samples_per_pixel
                
                # lehl@2022-02-10: Apply the softmax to the K changes per feature map, such
                # that we have no over- or underflows over time. 
                #
                # should hold: | alpha * K_change + (1 - alpha) * K | === 1
                #
                self.K *= (1 - self.alpha)
                self.K += self.alpha * K_change
                # del K_change

        # =======================================================================================
        #   CALCULATE OUTPUT
        # =======================================================================================
        #
        large_K = self.K.repeat(batch_size, 1, 1, 1, 1)
        large_K *= selected_multiplex_mask

        L = torch.zeros((batch_size, self.n*self.num_fm, self.fm_height, self.fm_width), device=self.device)

        for b in range(batch_size):
            L[b, ...] = F.conv2d(padded_A[b, ...].unsqueeze(0), minmax_on_fm(large_K[b,...].transpose(0,1)), padding=0) / self.num_fm

        output = torch.sum(L.reshape((batch_size, self.n, self.num_fm, L.shape[-2], L.shape[-1])), dim=1)

        # =======================================================================================
        #   UPDATE SCALING / MEAN ACTIVITY
        # =======================================================================================
        #
        if self.training and self.use_scaling:
            with torch.no_grad():
                # Perform "mean" history update
                # (including a scaling of the activity history)
                #
                self.M *= (1-self.beta)
                activity = (torch.sum(selected_multiplex_mask[:,:,:,0,0], dim=(0,2)) / batch_size)
                self.M += self.beta * (activity / (self.num_fm * self.n))

                # Scale cell strength
                self.S += self.gamma * (self.mu - self.M)
                self.S.data = torch.clip(self.S, min=0.5, max=2)

                if not wandb.run.disabled:
                    if ((self.iterations_trained % 50) == 0) and self.training:
                        wandb.log({
                            'lcl': {
                                'S': self.S.data.cpu().numpy(),
                                'M': self.M.data.cpu().numpy()
                            }
                        })

        # Keep track of number of training iterations
        self.iterations_trained += 1 if self.training else 0

        return output




class LaterallyConnectedLayer2(LaterallyConnectedLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        K = torch.zeros((self.num_fm, self.n, self.num_fm, self.n, self.k, self.k))
        self.register_parameter('K', torch.nn.Parameter(K, requires_grad=False))
        del K

        self.register_parameter('mu', torch.nn.Parameter(torch.ones((self.num_fm, self.n, self.fm_height, self.fm_width)) / self.n, requires_grad=False))
        self.register_parameter('S', torch.nn.Parameter(torch.ones(self.mu.shape), requires_grad=False))
        self.register_parameter('M', torch.nn.Parameter(torch.clone(self.mu), requires_grad=False))

    # Reshape the given activation to the compact multiplex shape
    # [b, fm, n, H, W] --> [b, fm*n, H, W]
    #
    def _activation_to_compact(self, A):
        As = A.shape
        return A.reshape((As[0], self.num_fm*self.n, As[3], As[4]))
    
    # Reshape the given activation to the extended multiplex shape
    # [b, fm, n, H, W] <-- [b, fm*n, H, W]
    #
    def _activation_from_compact(self, A):
        As = A.shape
        return A.reshape((As[0], self.num_fm, self.n, As[-2], As[-1]))

    # Reshape the kernel in the compact multiplex shape
    # [fmS, nS, fmT, nT, kH, kW] --> [fmS*nS, fmT*nT, kH, kW]
    #
    # The kernel self.K is stored in the extended multiplex shape
    #
    def _kernel_to_compact(self, K):
        Ks = K.shape
        return K.reshape((self.num_fm*self.n, self.num_fm*self.n, Ks[4], Ks[5]))

    # Reshape the kernel in the extended multiplex shape
    # [fmS, nS, fmT, nT, kH, kW] <-- [fmS*nS, fmT*nT, kH, kW]
    # 
    def _kernel_from_compact(self, K):
        Ks = K.shape
        return K.reshape((self.num_fm, self.n, self.num_fm, self.n, Ks[-2], Ks[-1]))


    def forward(self, A_input):
        if self.disabled:
            return torch.clone(A_input)

        with torch.no_grad():
            batch_size = A_input.shape[0]

            # Enlarge input for multipelx cells
            # --------------------------------------------------------------
            #
            # Turn [batch, feature_map, height, width]
            # into [batch, feature_map, **multiplex_cell**, height, width]
            #
            A = A_input.unsqueeze(2).repeat(1,1,self.n,1,1)

            # # # Feature map scaling
            # # # --------------------------------------------------------------
            # # #
            # scaling_factor = self.S.unsqueeze(0)
            # A *= scaling_factor

            # Calculate lateral activity
            # --------------------------------------------------------------
            #
            A_reshaped = self._activation_to_compact(A)
            K_reshaped = self._kernel_to_compact(torch.clone(self.K)).transpose(0,1)

            L = F.conv2d(A_reshaped, K_reshaped, padding=self.d.item())
            L = self._activation_from_compact(L)

            # Multiplex cell selection
            # --------------------------------------------------------------
            #
            L_idx = torch.argmax(L, dim=2, keepdim=True)

            Li = torch.argmax((1-self.iota)*A+self.iota*L, dim=2, keepdim=True)

            # self.A ->     [4, 20, 14, 14]
            # self.L ->     [4, 20, 14, 14]
            # fm_indices ->     [4, 5]
            # A_max ->      [4, 5]
            # filtered_A ->     [4, 5, 14, 14]
            #
            A_max = torch.argmax(torch.sum((1-self.iota)*A+self.iota*L, dim=(-2,-1)), dim=-1)
            fm_indices = A_max.shape[1] * A_max + torch.arange(0, A_max.shape[1]).to(self.device)
            del A_max
            # filtered_A = torch.stack([torch.index_select(i, dim=0, index=j) for i,j in zip(A_reshaped, fm_indices)])

            inactive_multiplex_idx = torch.zeros((batch_size, A_reshaped.shape[1] - fm_indices.shape[1]), device=self.device).to(torch.long)
            for b in range(batch_size):
                uniqs, cnts = torch.cat((torch.arange(A_reshaped.shape[1]).to(self.device), fm_indices[b,:])).unique(return_counts=True)
                inactive_multiplex_idx[b,:] = uniqs[cnts==1]

            # import code; code.interact(local=dict(globals(), **locals()))

            # Binary mask of which multiplex cells are active (per FM/pixel)
            multiplex_mask = torch.zeros(L.shape, device=self.device)
            multiplex_mask.scatter_(dim=2, index=L_idx, src=torch.ones(L_idx.shape, device=self.device))

            # Update Kernel
            # --------------------------------------------------------------
            #
            # A_reshaped = self._activation_to_compact(A)
            
            if self.training:
                K_change = torch.zeros(size=self._kernel_to_compact(self.K).shape, device=self.device)

                # filter_mask = multiplex_mask.reshape((batch_size, self.num_fm*self.n, multiplex_mask.shape[-2], multiplex_mask.shape[-1]))
                # A_filtered = A_reshaped * filter_mask

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
                        source_fms = A_reshaped[:, :, max(0,xoff_f):A_reshaped.shape[-2]+min(xoff_f,0), max(0,yoff_f):A_reshaped.shape[-1]+min(0,yoff_f)]
                        # source_fms = A_filtered[:, :, max(0,xoff_f):A_filtered.shape[-2]+min(xoff_f,0), max(0,yoff_f):A_filtered.shape[-1]+min(0,yoff_f)]
                        source_fms = source_fms.transpose(0,1)

                        xoff_i = self.d - x
                        yoff_i = self.d - y
                        target_fms = A_reshaped[:, :, max(0,xoff_i):A_reshaped.shape[-2]+min(xoff_i,0), max(0,yoff_i):A_reshaped.shape[-1]+min(0,yoff_i)]
                        target_fms = target_fms.transpose(0,1).unsqueeze(1)

                        # calculate the product of all feature maps (source) together with the
                        # to-be-influenced feature map (target) efficiently
                        #
                        # Initially, self.A contains [batch_size, num_fm, fm_height, fm_width] dimensions
                        #
                        # Designations of einsum characters:
                        #   a:  Extra dim to ensure each FM in target is multipled with the whole source blob
                        #   b:  Feature Map #
                        #   c:  Batch (see the transpose(0,1) calls)
                        #   d:  Feature Map Height
                        #   e:  Feature Map Width
                        #
                        tmp = torch.einsum('abcde,bcde->cab', target_fms, source_fms)

                        # # inhibit inactive multiplex cell changes
                        # #
                        # # binary mask of which multiplex cells are active, needs to be replicated to
                        # # incompass both source and target feature maps on the kernel change
                        # #
                        # # multiplex_mask: [batch, fms, multiplex]
                        # # mask: [batch, fms*multiplex, fms*multiplex]
                        # #
                        # mask = multiplex_mask[:,:,:,0,0]
                        # ms = mask.shape
                        # mask = mask.reshape((ms[0],ms[1]*ms[2])).repeat(1,ms[1]*ms[2]).reshape((ms[0],ms[1]*ms[2],ms[1]*ms[2]))
                        
                        # tmp = tmp * mask


                        # inhibit inactive multiplex cell changes
                        #
                        for b in range(batch_size):
                            # lehl@2022-04-18: Only source and target feature maps that are active
                            # should be changed/influenced by the changes calculated here
                            #
                            tmp[b, inactive_multiplex_idx[b,:], :] = 0
                            tmp[b, :, inactive_multiplex_idx[b,:]] = 0

                        # Average across the batch size
                        #
                        number_of_samples_per_pixel = torch.count_nonzero(tmp, dim=0)
                        number_of_samples_per_pixel[torch.where(number_of_samples_per_pixel == 0)] = 1
                        K_change[:, :, x, y] = torch.sum(tmp, dim=0) / number_of_samples_per_pixel
                        K_change[:, :, x, y] /= (tmp.shape[-1] * tmp.shape[-2])

                # import code; code.interact(local=dict(globals(), **locals()))

                self.K *= (1 - self.alpha)
                self.K += self.alpha * self._kernel_from_compact(K_change)
                del K_change

                # Update Feature Map Scaling
                # --------------------------------------------------------------
                #
                usage = torch.sum(multiplex_mask, dim=0) / multiplex_mask.shape[0]
              
                # import code; code.interact(local=dict(globals(), **locals()))
                self.M *= (1 - self.beta)
                self.M += self.beta * usage
                self.S = nn.Parameter(torch.clip(self.S + self.gamma * (self.mu - self.M), min=0.5, max=2.0))
                

            if self.iterations_trained % 100 == 0:
                np.set_printoptions(precision=2, suppress=True)
                print("USAGE\n",torch.sum(multiplex_mask, dim=(0,-1,-2)).cpu().detach().numpy())
                print("MEAN\n", torch.sum(self.M, dim=(-2,-1)).cpu().detach().numpy() / (14*14))
                print("STRENGTH\n", torch.sum(self.S,dim=(-2,-1)).cpu().detach().numpy() / (14*14))
            # import code; code.interact(local=dict(globals(), **locals()))

        # Calculate output using only the active multiplex cells
        # --------------------------------------------------------------
        #
        A_reshaped = self._activation_to_compact(A)

        L = F.conv2d(A_reshaped, minmax_on_fm(self._kernel_to_compact(self.K).transpose(0,1)), padding=self.d.item())
        L = self._activation_from_compact(L)
        
        self.iterations_trained += 1 if self.training else 0
        # import code; code.interact(local=dict(globals(), **locals()))

        output = torch.sum(L * multiplex_mask, dim=2)
        return output


