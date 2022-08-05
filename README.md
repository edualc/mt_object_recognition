# mt_object_recognition

This is the repository for the code used in my master thesis, titled "Leveraging Neuroscience for Deep Learning Based Object Recognition".


### Installation

Run the makefile with ```make init```. After that the environment will be created and necessary dependencies are installed.

The code is based on Python version ```3.8.10```.

When you later just want to run the environment, use the ```scripts/activate_env.sh``` shell file. For using Jupyter, after activating the environment run ```scrips/start_jupyter.sh```.

The Optuna hyperparameter tuning library uses SQLite for synchronization of parallel trial runs, if you intend to use it you might have to install SQLite on your machine: ```sudo apt-get install sqlite```.

### Comments on the repository structure (taken from thesis)

The code found in the mt_object_recognition GitHub repository includes all code used for the experiments of this thesis and features the implementation of the laterally connected layer (LCL). The necessary information to install and run the dependencies can be found in the README.md file, located in the base directory. Additionally, the base directory contains many of the python scripts used to run experiments or longer running tasks, often remotely on a GPU cluster. The provided Dockerfile allows you to create a Docker container with all the necessary dependencies to run the code remotely. The other directories are explained below in more detail:

- **_old** This directory contains experimental, but finally unused code, such as part of the implementation of the ["Texture Synthesis using CNN" paper](https://arxiv.org/abs/1505.07376) and older iPython notebooks. We do not recommend spending further time here. experiment_results This directory hosts the many CSV files from our experiments with class predictions and results across the various datasets (primarily MNIST-C).

- **images** A few samples images, as well as all datasets can be found here. Note that the datasets are not directly provided in the GitHub repository, but rather expected to be downloaded and put into this directory. For MNIST, you can use the torchvision library to automatically download it for you (into the ```mnist``` directory). For MNIST-C, please use the static dataset provided by the paper authors on their [GitHub repository](https://github.com/google-research/mnist-c) and store it in the ```mnist_c``` directory. For example, the MNIST-C variant dotted_line is expected to be found at path ```mt_object_recognition/images/mnist_c/dotted_line/train_images.npy```.

- **lateral_connections** The main implementation code for the LCL is found in this directory. The ```character_models.py``` file hosts a variety of models with and without using the LCL. dataset.py implements different datasets that were largely used for debugging purposes, but not in the most recent version of ablation studies. ```lateral_model.py``` implements an older variant of an integrated LCL, but is not used anymore. ```layers.py``` contains the LCL implementation, of which there are three variants (```LaterallyConnectedLayer```, ```LaterallyConnectedLayer2``` and ```LaterallyConnectedLayer3```). Please use ```LaterallyConnectedLayer3```, as it is the most recent version and implements the architecture described in this thesis. Versions one and two implement an inferior set of characteristics, such as a scaling mechanism and multiplex selection on a per-pixel basis (rather than per feature map). The ```loaders.py``` file implements utility methods for quickly loading the datasets as PyTorch loaders. Similarly, ```model_factory.py``` implements methods to quickly load a model with default configurations and even loads the checkpoint weights, if a path is provided. ```torch_utils.py``` features a variety of methods that were used to apply min-max scaling and softmax on a feature map level, rather than across the whole data cube.

- **models** All the network checkpoints are stored in this folder. This also includes pretrained checkpoints that were used for fine-tuning or re-training the networks with an added LCL. It is currently not planned that we upload our model checkpoints, as they can easily be recreated locally with the training scripts.

- **notebooks** All the iPython notebooks are stored here, most of which should be visible through GitHub already (though just rendered without interactivity in mind.) All notebooks with the ```Ablation__``` prefix include ablation studies and debugging scenarios to test, whether the LCL performs as it should.

- **scripts** Here you can find helper scripts to quickly activate the Python environment, start Jupyter or check the GPU usage. ```dgx_get_gpu.sh``` is an example call for running this code on the ZHAW GPU cluster infrastructure.

The experiments of this thesis outlined in Chapter 4 using the hyperparameter search with Optuna can be recreated using the scripts found in the base directory and are marked with the prefix ```optuna_```. Any future changes to the repository will be marked in the README.md file, please check here first in case of differences.
