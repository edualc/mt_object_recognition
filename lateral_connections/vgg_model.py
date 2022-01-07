import pickle
import torch
from typing import List


class VggModel():
    def __init__(self,
                 model_path: str,
                 device: torch.device, 
                 important_layers: List[str] = [
                    'relu1_1', 'pool1', 'pool2', 'pool3', 'pool4'
                 ]):
        
        self.net = self.load_model(model_path)
        self.device = device
        self.important_layers = important_layers

        # Make sure that the network is moved to the GPU, if available
        if torch.cuda.is_available():
            self.net = self.net.to(self.device)

        self._remove_unnecessary_layers()

    # The VGG19 model has additional layers which can be removed,
    # as only the output of the final pool layer is relevant
    #
    def _remove_unnecessary_layers(self):
        i = 0
        for name, layer in self.net.named_children():
            if name == self.important_layers[-1]:
                break
            i += 1
        self.net = self.net[:(i+1)]

    # Loads the pre-trained model. Based on the implementation of 
    # https://github.com/honzukka/texture-synthesis-pytorch/blob/master/utilities.py#L43
    #
    # (In case of errors when running on CPU, try their monkey patch)
    #
    def load_model(self, model_path):
        modules, model_pickle = pickle.load(open(model_path, 'rb'))
        model = pickle.loads(model_pickle)

        return model
