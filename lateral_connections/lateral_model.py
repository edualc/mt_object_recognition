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

# Arguments:
#    - vgg_model:               An instance of the VGG19 model, pre-trained
#    - num_output_repetitions:  How many times the output of the VGG19 network will be
#                                   tiled to give more room for the netfragments to form
#    - horizon_length:          How many time steps should be kept/considered for the
#                                   mean cell activation measurements
#    - mean_cell_activation:    How much a single cell should be active
#    - gram_distance:           How many pixels should be considered as neighborhood
#    - delta:                   How much the cell strength will be corrected, given by
#                                   the difference in overall cell activation compared
#                                   to the mean cell activation
#
class OldLateralModel():
    def __init__(self, vgg_model: VggModel, num_output_repetitions: int = 2,
        horizon_length: int = 200, mean_cell_activation: float = 0.01,
        delta: float = 0.01, gram_distance: int = 4, distance_map_template: DistanceMapTemplate = ManhattanDistanceTemplate):        
        
        self.vgg_model = vgg_model
        self.gpu_device = self.vgg_model.device
        self.num_output_repetitions = num_output_repetitions
        self.horizon_length = horizon_length
        self.mean_cell_activation = mean_cell_activation
        self.gram_distance = gram_distance
        self.delta = delta
        self.distance_map_template = distance_map_template

        self.dtype = 'float32'
        self._reset_model()

    def _reset_model(self):
        self.C = None               # Enlarged Cell Activations
        self.O = None               # Output
        self.G_d = None             # Distanced Gram-Matrix
        self.S = None               # Cell Strength
        self.H = None               # Mean Output over time (-> horizon_length)
        self.distance_map = None    # Neighbourhood Distance Map
        self.L = None               # Lateral Connection Strength

    # Ensure, that the necessary initializations take place. However, they
    # are dependent on the overall shape of the image(s)
    #
    # - self.S:                 Randomized cell strength for each cell
    # - self.H:                 Pre-computed mean output, updated with weighted sum
    # - self.distance_map:      Distance map to use for neighbourhood calculations
    #
    def _initialize(self, x):
        if self.S is None:
            self.S = torch.rand(*((self.num_output_repetitions,) + x.shape)).float().to(self.gpu_device)

        if self.H is None:
            self.H = torch.mul(
                        self.mean_cell_activation,
                        torch.ones(size=(self.num_output_repetitions,) + x.shape)
                    ).float().to(self.gpu_device)

        if self.distance_map is None:
            self.distance_map = torch.from_numpy(self.distance_map_template.generate_distance_map(x, self.gram_distance)) \
                                     .float().to(self.gpu_device)

    # Multiply the given :out activations with the :distance_map, such that
    # the resulting output :new includes all the neighbourhood influences,
    # given the choice of :distance_map.
    #
    def distanced_gram_matrix(self, out, distance_map):
        new = torch.empty((out.shape[0],) + distance_map.shape[:2]).float().to(self.gpu_device)
        for i in range(out.shape[0]):
            new[i, :, :] = torch.sum(out[i, :, :] * distance_map, axis=(2,3))
        return new

    def forward(self, img, update=True):
        with torch.no_grad():
            img_gpu = img.to(self.gpu_device)
            x = self.vgg_model.net(img_gpu)[0, ...]

            self._initialize(x)

            self.forward__calculation_step(x)

            if update:
                self.forward__update_step(x)

            return self.O

    # 1. CALCULATION STEP
    def forward__calculation_step(self, x):    
        # G_d = "x * distance_map"
        self.G_d = self.distanced_gram_matrix(x, self.distance_map)
        
        # Enlarge C by :num_output_repetitions to reduce crosstalk activations
        # and give more space for selforganization to go out of its way
        #
        self.C = torch.broadcast_to(x, (self.num_output_repetitions,) + x.shape).float()

        # L = G_d * S
        self.L = torch.matmul(self.G_d, self.S)

        # O = C * L
        self.O = self.C * self.L

    # 2. UPDATE STEP
    def forward__update_step(self, x):
        # O and S are clipped, to stay within expected boundaries
        # lehl@2022-01-07: TODO - Would Sigmoid/TanH make sense instead of clipping?

        # factor = 1 / horizon
        factor = torch.Tensor([1 / self.horizon_length]).float().to(self.gpu_device)

        # H = factor * O + (1 - factor) * H
        self.H = torch.mul(factor, torch.clip(self.O, min=0.0, max=1.0)) + torch.mul(torch.subtract(torch.Tensor([1]).to(self.gpu_device), factor), self.H)

        # S = S - delta * (H - mean_cell_activation)
        self.S = torch.subtract(self.S, torch.mul(self.delta, torch.subtract(self.H, self.mean_cell_activation)))
        self.S = torch.clip(self.S, min=-1.0, max=1.0)

class NewLateralModel:
    def __init__(self, vgg_model, num_output_repetitions=4, horizon_length=200, mean_cell_activation=0.01, distance=1, alpha=0.01):
        self.vgg_model = vgg_model
        self.gpu_device = self.vgg_model.device
        self.num_output_repetitions = num_output_repetitions
        self.horizon_length = horizon_length
        self.mean_cell_activation = mean_cell_activation
        self.distance = distance

        self.epoch = 0
        self.alpha = alpha

        self.lateral_strength = None
        
    def forward(self, img, update=True):
        with torch.no_grad():
            img_gpu = img.to(self.gpu_device)
            x = self.vgg_model.net(img_gpu)[0, ...]

            x = x.cpu().detach().numpy()
            
            self.forward__calculation_step(x)

            if update:
                self.forward__update_step(x)

            self.epoch += 1
            return self.O
        
    def forward__calculation_step(self, activations):
        num_featuremaps, fm_height, fm_width = activations.shape

        # laterals = np.zeros(shape=activations.shape)
        kernel_size = 2*self.distance+1

        if self.lateral_strength is None:
            self.lateral_strength = 1e-2 * np.ones(shape=activations.shape + (num_featuremaps, kernel_size, kernel_size), dtype=float)

        from tqdm import tqdm

        # Reduce lateral_strength for weighted average
        self.lateral_strength *= (1 - self.alpha)

        # Prepare output
        self.O = np.zeros(shape=activations.shape, dtype=float)
        self.S = np.zeros(shape=activations.shape, dtype=float)

        for fm_a in tqdm(range(num_featuremaps), leave=False):
            for i in range(fm_height):
                for j in range(fm_width):

                    # TODO: Padding Iteration
                    range_x_min = max(0, i - self.distance)
                    range_x_max = min(fm_height - 1, i + self.distance + 1)
                    range_y_min = max(0, j - self.distance)
                    range_y_max = min(fm_width - 1, j + self.distance + 1)

                    # A = activations[fm_a, range_x_min:range_x_max, range_y_min:range_y_max]
                    A = activations[fm_a, i, j]
                    B = activations[:, range_x_min:range_x_max, range_y_min:range_y_max]

                    if B.shape[1] < kernel_size:
                        if i - self.distance < 0:
                            # CASE: at top edge
                            # tmp = np.zeros((kernel_size, A.shape[1]))
                            # tmp[1:, :] = A
                            # A = tmp

                            tmp = np.zeros((num_featuremaps, kernel_size, B.shape[2]))
                            tmp[:, 1:, :] = B
                            B = tmp

                        elif i + self.distance + 1 > fm_height - 1:
                            # CASE: at bottom edge
                            # tmp = np.zeros((kernel_size, A.shape[1]))
                            # tmp[:-1, :] = A
                            # A = tmp

                            tmp = np.zeros((num_featuremaps, kernel_size, B.shape[2]))
                            tmp[:, :-1, :] = B
                            B = tmp

                        else:
                            print("ERROR i bad")
                            import code; code.interact(local=dict(globals(), **locals()))

                    if B.shape[2] < kernel_size:
                        if j - self.distance < 0:
                            # CASE: at left edge
                            # tmp = np.zeros((kernel_size, kernel_size))
                            # tmp[:, 1:] = A
                            # A = tmp

                            tmp = np.zeros((num_featuremaps, kernel_size, kernel_size))
                            tmp[:, :, 1:] = B
                            B = tmp

                        elif j + self.distance + 1 > fm_width - 1:
                            # CASE: at right edge
                            # tmp = np.zeros((kernel_size, kernel_size))
                            # tmp[:, :-1] = A
                            # A = tmp

                            tmp = np.zeros((num_featuremaps, kernel_size, kernel_size))
                            tmp[:, :, :-1] = B
                            B = tmp

                        else:
                            print("ERROR j bad")
                            import code; code.interact(local=dict(globals(), **locals()))


                    # import code; code.interact(local=dict(globals(), **locals()))

                    # Add lateral_strength of the given sample
                    self.lateral_strength[fm_a, i, j] += self.alpha * np.clip(A*B, a_min=0.0, a_max=1.0)

                    self.S[fm_a, i, j] = np.sum(self.lateral_strength[fm_a, i, j] * B)


                    # # TODO: Padding Iteration
                    # range_x_min = max(0, x - self.distance)
                    # range_x_max = min(fm_height - 1, x + self.distance + 1)
                    # range_y_min = max(0, y - self.distance)
                    # range_y_max = min(fm_width - 1, y + self.distance + 1)

                    # a = activations[fm_a, range_x_min:range_x_max, range_y_min:range_y_max]
                    # b = activations[fm_b, range_x_min:range_x_max, range_y_min:range_y_max]
                    
                    # if (a.shape == (5,5)) and (b.shape == (5,5)):
                    #     import code; code.interact(local=dict(globals(), **locals()))


                    # laterals[fm_a, x, y] += np.sum(a*b)
                        

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(laterals[fm_a, :, :], vmin=0, vmax=laterals[fm_a, :, :].max())
            # plt.savefig(f"{fm_a}.png")
            # plt.close()

        self.O = activations * self.S

        # import code; code.interact(local=dict(globals(), **locals()))

        num_fms_to_plot = 8
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, num_fms_to_plot, figsize=(4 * num_fms_to_plot, 4 * 3))

        for i in range(num_fms_to_plot):
            axs[0, i].imshow(activations[i, :, :], vmin=0, vmax=1)
            axs[0, i].set_title(f"[FM #{i}] Activation (vmax={int(np.max(activations[i,:,:]))})")
            axs[1, i].imshow(self.S[i, :, :], vmin=0, vmax=np.max(self.S[i, :, :]))
            axs[1, i].set_title(f"[FM #{i}] Lateral Connections (vmax={int(np.max(self.S[i,:,:]))})")
            axs[2, i].imshow(self.O[i, :, :])
            axs[2, i].set_title(f"[FM #{i}] Generated Output (vmax={int(np.max(self.O[i,:,:]))})")
        plt.tight_layout()
        plt.savefig(f"epoch_images/new_lateral_model/epoch{self.epoch}.jpg")
        plt.close()

        # import code; code.interact(local=dict(globals(), **locals()))

    def forward__update_step(self, x):
        pass

class NewerLateralModel:
    def __init__(self, vgg_model, num_output_repetitions=4, horizon_length=200, mean_cell_activation=0.01, distance=1, alpha=0.1):
        self.vgg_model = vgg_model
        self.gpu_device = self.vgg_model.device
        self.num_output_repetitions = num_output_repetitions
        self.horizon_length = horizon_length
        self.mean_cell_activation = mean_cell_activation
        self.distance = distance

        self.epoch = 0
        self.alpha = alpha

        self.lateral_strength = None
        
    def forward(self, img, update=True):
        with torch.no_grad():
            img_gpu = img.to(self.gpu_device)
            x = self.vgg_model.net(img_gpu)[0, ...]

            x = x.cpu().detach().numpy()
            
            self.forward__calculation_step(x)

            if update:
                self.forward__update_step(x)

            self.epoch += 1
            return self.O
        
    def forward__calculation_step(self, activations):
        num_featuremaps, fm_height, fm_width = activations.shape

        # laterals = np.zeros(shape=activations.shape)
        kernel_size = 2*self.distance+1

        if self.lateral_strength is None:
            self.lateral_strength = 1e-2 * np.ones(shape=activations.shape + (num_featuremaps, kernel_size, kernel_size), dtype=float)

        from tqdm import tqdm

        # Reduce lateral_strength for weighted average
        self.lateral_strength *= (1 - self.alpha)

        #
        #   PyTorch ConvNet2D Demystified
        #   https://engineering.purdue.edu/DeepLearn/pdf-kak/week6.pdf
        #   ====================================================================================================
        #

        # Prepare output
        self.O = np.zeros(shape=activations.shape, dtype=float)
        self.S = np.zeros(shape=activations.shape, dtype=float)
        self.O2 = np.zeros(shape=activations.shape, dtype=float)
        self.S2 = np.zeros(shape=activations.shape, dtype=float)

        for fm_a in tqdm(range(num_featuremaps), leave=False):
            for i in range(fm_height):
                for j in range(fm_width):

                    # TODO: Padding Iteration
                    range_x_min = max(0, i - self.distance)
                    range_x_max = min(fm_height - 1, i + self.distance + 1)
                    range_y_min = max(0, j - self.distance)
                    range_y_max = min(fm_width - 1, j + self.distance + 1)

                    # A = activations[fm_a, range_x_min:range_x_max, range_y_min:range_y_max]
                    A = activations[fm_a, i, j]
                    B = activations[:, range_x_min:range_x_max, range_y_min:range_y_max]

                    if B.shape[1] < kernel_size:
                        if i - self.distance < 0:
                            # CASE: at top edge
                            tmp = np.zeros((num_featuremaps, kernel_size, B.shape[2]))
                            tmp[:, 1:, :] = B
                            B = tmp

                        elif i + self.distance + 1 > fm_height - 1:
                            # CASE: at bottom edge
                            tmp = np.zeros((num_featuremaps, kernel_size, B.shape[2]))
                            tmp[:, :-1, :] = B
                            B = tmp

                        else:
                            print("ERROR i bad")
                            import code; code.interact(local=dict(globals(), **locals()))

                    if B.shape[2] < kernel_size:
                        if j - self.distance < 0:
                            # CASE: at left edge
                            tmp = np.zeros((num_featuremaps, kernel_size, kernel_size))
                            tmp[:, :, 1:] = B
                            B = tmp

                        elif j + self.distance + 1 > fm_width - 1:
                            # CASE: at right edge
                            tmp = np.zeros((num_featuremaps, kernel_size, kernel_size))
                            tmp[:, :, :-1] = B
                            B = tmp

                        else:
                            print("ERROR j bad")
                            import code; code.interact(local=dict(globals(), **locals()))

                    # Add lateral_strength of the given sample
                    self.lateral_strength[fm_a, i, j] += self.alpha * np.clip(A*B, a_min=0.0, a_max=1.0)

                    self.S[fm_a, i, j] = np.sum(self.lateral_strength[fm_a, i, j] * B)


        self.l2 = np.copy(self.lateral_strength)
        self.l2 = np.mean(self.lateral_strength, axis=(0,1,2))


        # for fm_a in tqdm(range(num_featuremaps), leave=False):
        #     for i in range(fm_height):
        #         for j in range(fm_width):

        #             self.S2[fm_a, i, j] = 


        self.O2 = activations * self.S2
        import code; code.interact(local=dict(globals(), **locals()))

        self.O = activations * self.S

        # num_fms_to_plot = 8
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(3, num_fms_to_plot, figsize=(4 * num_fms_to_plot, 4 * 3))

        # for i in range(num_fms_to_plot):
        #     axs[0, i].imshow(activations[i, :, :], vmin=0, vmax=1)
        #     axs[0, i].set_title(f"[FM #{i}] Activation (vmax={int(np.max(activations[i,:,:]))})")
        #     axs[1, i].imshow(self.S[i, :, :], vmin=0, vmax=np.max(self.S[i, :, :]))
        #     axs[1, i].set_title(f"[FM #{i}] Lateral Connections (vmax={int(np.max(self.S[i,:,:]))})")
        #     axs[2, i].imshow(self.O[i, :, :])
        #     axs[2, i].set_title(f"[FM #{i}] Generated Output (vmax={int(np.max(self.O[i,:,:]))})")
        # plt.tight_layout()
        # plt.savefig(f"epoch_images/newer_lateral_model/epoch{self.epoch}.jpg")
        # plt.close()

        # import code; code.interact(local=dict(globals(), **locals()))

    def forward__update_step(self, x):
        pass


class LateralModel:
    def __init__(self, vgg_model, num_output_repetitions=2,
        horizon_length=200, expected_duty_cycle=0.005, distance=1, alpha=0.1):
       
        self.vgg_model = vgg_model
        self.gpu_device = self.vgg_model.device

        self.n = num_output_repetitions    # number of feature cell copies to reduce cross talk
        self.d = distance                  # distance
        self.h = horizon_length            # horizon length
        self.k = 2 * self.d + 1            # kernel size
        self.mu = expected_duty_cycle      # expected duty cycle
        self.alpha = alpha                 # alpha (kernel learning rate)
        
        self.iterations_trained = 0

        self.O = None
        self.K = None
        self.H = None
        self.S = None
        self.L = None
        
    def save_model(self, file_path:str=None):
        if not file_path:
            file_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.h5'

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
    def forward(self, img, update=True):
        with torch.no_grad():
            img_gpu = img.to(self.gpu_device)
            
            # lehl2022-01-17:
            # TODO: Add Threshold Clipping?
            #
            A = self.vgg_model.net(img_gpu)[0, ...]
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
                            self.K[:, a, x, y] += self.alpha * torch.sum(self.A[:, x:, y:] * self.A[a, :self.A.shape[-2]-x, :self.A.shape[-1]-y], dim=(1,2))
            
            # Applying the Kernal Convolution
            #
            self.L = F.conv2d(self.A.unsqueeze_(0), self.K.transpose_(0, 1), padding=self.d).squeeze()

            if update:
                # Update of Cell Activity History
                #
                h_factor = torch.divide(torch.Tensor([1]), torch.Tensor([self.h])).to(self.gpu_device)

                self.H = h_factor * self.L + (1 - h_factor) * self.H

                # Cell Activity Normalization
                #
                self.S = self.mu - self.H

                # Keep track of number of training iterations
                self.iterations_trained += 1

            # Activation Scaling
            #
            self.O = self.A * self.L * self.S

            return self.O
