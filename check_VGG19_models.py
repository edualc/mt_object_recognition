import os
import pandas as pd
from lateral_connections import VggWithLCL
from lateral_connections.loaders import get_loaders

model_configs = {
    '2022-04-01_181849': { 'use_lcl': False,   'dropout': 0.0, 'learning_rate': 0.003 },
    '2022-04-01_193339': { 'use_lcl': False,   'dropout': 0.2, 'learning_rate': 0.003 },
    '2022-04-01_204931': { 'use_lcl': False,   'dropout': 0.5, 'learning_rate': 0.003 },
    '2022-04-01_220539': { 'use_lcl': False,   'dropout': 0.2, 'learning_rate': 0.003 },
    '2022-04-01_232146': { 'use_lcl': False,   'dropout': 0.2, 'learning_rate': 0.0003 },
    '2022-04-02_003615': { 'use_lcl': False,   'dropout': 0.2, 'learning_rate': 0.03 },
    '2022-04-02_014927': { 'use_lcl': False,   'dropout': 0.2, 'learning_rate': 0.001 },
    '2022-04-02_030457': { 'use_lcl': False,   'dropout': 0.2, 'learning_rate': 0.0001 },
}

files = []
for file in os.listdir('models/vgg_with_lcl/'):
    if file == '.gitkeep':
        continue
    if not os.path.isdir('models/vgg_with_lcl/' + file):
        files.append(file)

eval_data = dict()

for file in files:
    file_path = 'models/vgg_with_lcl/' + file
    model_ident = file.split('__')[0]

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
    model.load(file_path)

    if config['use_lcl']:
        model.features.lcl3.enable()

    _, _, test_loader, _ = get_loaders(config['batch_size'])
    acc, loss = model.test(test_loader)

    iterations, epoch = file.split('__')[1].split('.')[0].split('_')
    eval_data[file] = {
        'epoch': epoch[1:],
        'iterations': iterations[2:],
        'test_acc': acc,
        'test_loss': loss,
        'dropout': model_configs[model_ident]['dropout'],
        'learning_rate': model_configs[model_ident]['learning_rate']
    }
    print(eval_data[file])

df = pd.DataFrame(eval_data).transpose()
df['model'] = [x.split('__')[0] for x in df.index]
df.to_csv('check_models__VGG19.csv')
