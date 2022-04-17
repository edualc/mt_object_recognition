import os
import pandas as pd
import torch
from lateral_connections import VggWithLCL
from lateral_connections.loaders import get_loaders, load_mnistc
from tqdm import tqdm

VGG19_TRAINED_MODEL = 'models/vgg_with_lcl/VGG19__2022-04-02_014927__it13999_e2.pt'

model_configs = {
    # '2022-04-01_181849': { 'use_lcl': False, 'dropout': 0.0, 'learning_rate': 0.003 },
    # '2022-04-01_193339': { 'use_lcl': False, 'dropout': 0.2, 'learning_rate': 0.003 },
    # '2022-04-01_204931': { 'use_lcl': False, 'dropout': 0.5, 'learning_rate': 0.003 },
    # '2022-04-01_220539': { 'use_lcl': False, 'dropout': 0.2, 'learning_rate': 0.003 },
    # '2022-04-01_232146': { 'use_lcl': False, 'dropout': 0.2, 'learning_rate': 0.0003 },
    # '2022-04-02_003615': { 'use_lcl': False, 'dropout': 0.2, 'learning_rate': 0.03 },
    '2022-04-02_014927': { 'use_lcl': False, 'dropout': 0.2, 'learning_rate': 0.001 },
    # '2022-04-02_030457': { 'use_lcl': False, 'dropout': 0.2, 'learning_rate': 0.0001 },
}

mnist_c_datasets = []

folder_path = 'images/mnist_c/'
for file in os.listdir(folder_path):
    dataset_path = folder_path + file

    if os.path.isdir(dataset_path):
        mnist_c_datasets.append(file)

model_ident = '2022-04-02_014927'
config = {
    'num_classes': 10,
    'learning_rate': model_configs[model_ident]['learning_rate'],
    'dropout': model_configs[model_ident]['dropout'],
    'num_multiplex': 4,
    'batch_size': 10,
    'use_lcl': model_configs[model_ident]['use_lcl'],
    'num_epochs': 5
}
model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'], num_multiplex=config['num_multiplex'], do_wandb=False)
model.load(VGG19_TRAINED_MODEL)

eval_data = dict()

pbar = tqdm(mnist_c_datasets, leave=False)
for dirname in pbar:
    pbar.set_description(f"Processing MNIST-C Dataset '{dirname}'...")
    dataset = load_mnistc(dirname, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1, pin_memory=True)

    acc, loss = model.test(loader)

    eval_data[dirname] = {
        'acc': acc,
        'loss': loss
    }

try:
    df = pd.DataFrame(eval_data).transpose()
    df['corruption'] = df.index
    df = df.sort_values('acc')
    df.to_csv('check_VGG19_corruptions.csv')
except Exception:
    import code; code.interact(local=dict(globals(), **locals()))

import code; code.interact(local=dict(globals(), **locals()))
