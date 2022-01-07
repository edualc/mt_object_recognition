import numpy as np
import torch

from .vgg_model import VggModel
from .dataset import CustomImageDataset
from .distance_map import DistanceMapTemplate, ManhattanDistanceTemplate

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
class LateralModel():
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
        self.C = None               # Enlarged cell activations
        self.O = None               # Output
        self.G_d = None             # Distanced Gram-Matrix
        self.S = None               # Cell Strength
        self.H = None               # Mean Output over time (-> horizon_length)
        self.distance_map = None    # Neighbourhood distance map

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

            # CALCULATION STEP
            # ================================================================================
            
            # G_d = "x * distance_map"
            self.G_d = self.distanced_gram_matrix(x, self.distance_map)
            
            # Enlarge C by :num_output_repetitions to reduce crosstalk activations
            # and give more space for selforganization to go out of its way
            #
            self.C = torch.broadcast_to(x, (self.num_output_repetitions,) + x.shape).float()

            # L = G_d * S
            L = torch.matmul(self.G_d, self.S)

            # O = C * L
            self.O = self.C * L

            if update:
                # UPDATE STEP
                # ================================================================================
                # O and S are clipped, to stay within expected boundaries
                # lehl@2022-01-07: TODO - Would Sigmoid/TanH make sense instead of clipping?

                # factor = 1 / horizon
                factor = torch.Tensor([1 / self.horizon_length]).float().to(self.gpu_device)

                # H = factor * O + (1 - factor) * H
                self.H = torch.mul(factor, torch.clip(self.O, min=0.0, max=1.0)) + torch.mul(torch.subtract(torch.Tensor([1]).to(self.gpu_device), factor), self.H)

                # S = S - delta * (H - mean_cell_activation)
                self.S = torch.subtract(self.S, torch.mul(self.delta, torch.subtract(self.H, self.mean_cell_activation)))
                self.S = torch.clip(self.S, min=-1.0, max=1.0)

            return self.O
