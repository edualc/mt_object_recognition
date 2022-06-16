from collections import OrderedDict
from .lateral_model import *
from .character_models import *

CONFIGS = {
    'vggonly': {
        'num_classes': 10,
        'learning_rate': 3e-4,
        'dropout': 0.2,
        'num_epochs': 4,
        'batch_size': 10,
        'use_lcl': False,
        'num_multiplex': 4,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.0,
        'lcl_iota': 0.2
    },
    'vgg_full': {
        'num_classes': 10,
        'batch_size': 10,
        'learning_rate': 3e-4,
    },
    'tiny_cnn': {
        'batch_size': 10,
        'conv_channels': 10,
        'learning_rate': 3e-4,
        'num_classes': 10,
        'run_identifier': '',
    },
    'tiny_lateral_net': {
        'num_multiplex': 4,
        'lcl_distance': 2,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.0,
        'lcl_iota': 0.2,
        'batch_size': 10,
        'conv_channels': 10,
        'learning_rate': 3e-4,
        'num_classes': 10,
        'run_identifier': '',
    },
    'lcl': {
        'num_classes': 10,
        'learning_rate': 1e-3,
        'dropout': 0.2,
        'num_epochs': 4,
        'batch_size': 10,
        'use_lcl': True,
        'num_multiplex': 4,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.0,
        'lcl_iota': 0.2
    },
    'vgg16_lcl': {
        'num_classes': 10,
        'learning_rate': 1e-3,
        'dropout': 0.2,
        'num_epochs': 4,
        'batch_size': 10,
        'use_lcl': True,
        'num_multiplex': 4,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.0,
        'lcl_iota': 0.2
    },
    'vgg19r_lcl': {
        'num_classes': 10,
        'learning_rate': 1e-4,
        'num_multiplex': 4,
        'batch_size': 10,
        'num_epochs': 5,
        'lcl_alpha': 3e-4,
        'lcl_eta': 0.01,
        'lcl_theta': 0.2,
        'lcl_iota': 0.2,
        'lcl_distance': 2,
        'lcl_k': 5,
        'after_pooling': 5,
        'use_scaling': False,
        'random_k_change': False,
        'random_multiplex_selection': False,
        'gradient_learn_k': False,
        'fc_only': False,
    },
    'vgg19r_lcl__scaling': {
        'num_classes': 10,
        'learning_rate': 1e-4,
        'num_multiplex': 4,
        'batch_size': 10,
        'num_epochs': 5,
        'lcl_alpha': 3e-4,
        'lcl_eta': 0.01,
        'lcl_theta': 0.2,
        'lcl_iota': 0.2,
        'lcl_distance': 2,
        'lcl_k': 5,
        'after_pooling': 5,
        'use_scaling': True,
        'random_k_change': False,
        'random_multiplex_selection': False,
        'gradient_learn_k': False,
        'fc_only': False,
    },
    'vgg19r_lcl__random_k_change': {
        'num_classes': 10,
        'learning_rate': 1e-4,
        'num_multiplex': 4,
        'batch_size': 10,
        'num_epochs': 5,
        'lcl_alpha': 3e-4,
        'lcl_eta': 0.01,
        'lcl_theta': 0.2,
        'lcl_iota': 0.2,
        'lcl_distance': 2,
        'lcl_k': 5,
        'after_pooling': 5,
        'use_scaling': False,
        'random_k_change': True,
        'random_multiplex_selection': False,
        'gradient_learn_k': False,
        'fc_only': False,
    },
    'vgg19r_lcl__random_multiplex_selection': {
        'num_classes': 10,
        'learning_rate': 1e-4,
        'num_multiplex': 4,
        'batch_size': 10,
        'num_epochs': 5,
        'lcl_alpha': 3e-4,
        'lcl_eta': 0.01,
        'lcl_theta': 0.2,
        'lcl_iota': 0.2,
        'lcl_distance': 2,
        'lcl_k': 5,
        'after_pooling': 5,
        'use_scaling': False,
        'random_k_change': False,
        'random_multiplex_selection': True,
        'gradient_learn_k': False,
        'fc_only': False,
    },
    'vgg19r_lcl__fc_only': {
        'num_classes': 10,
        'learning_rate': 1e-4,
        'num_multiplex': 4,
        'batch_size': 10,
        'num_epochs': 5,
        'lcl_alpha': 3e-4,
        'lcl_eta': 0.01,
        'lcl_theta': 0.2,
        'lcl_iota': 0.2,
        'lcl_distance': 2,
        'lcl_k': 5,
        'after_pooling': 5,
        'use_scaling': False,
        'random_k_change': False,
        'random_multiplex_selection': False,
        'gradient_learn_k': False,
        'fc_only': True,
    },
}

MODEL_CLASS = {
    'vggonly': VggWithLCL,
    'vgg_full': VggFull,
    'vgg16_lcl': SmallVggWithLCL,
    'lcl': VggWithLCL,
    'tiny_cnn': TinyCNN,
    'vgg19r_lcl': VGGReconstructionLCL,
    'vgg19r_lcl__random_k_change': VGGReconstructionLCL,
    'vgg19r_lcl__random_multiplex_selection': VGGReconstructionLCL,
    'vgg19r_lcl__fc_only': VGGReconstructionLCL,
}

def get_config_by_key(model_key):
    if model_key not in CONFIGS.keys():
        raise ValueError(':model_key not found, available models: ', CONFIGS.keys())

    return CONFIGS[model_key]

def load_model_by_key(model_key, model_path=None, config=None):
    if model_key not in CONFIGS.keys():
        raise ValueError(':model_key not found, available models: ', CONFIGS.keys())

    if config is None:
        config = CONFIGS[model_key]

    if model_key == 'vggonly':
        model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
            num_multiplex=config['num_multiplex'], do_wandb=False, run_identifier="",
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        if model_path is not None:
            model.load(model_path)
        return model    

    elif model_key == 'vgg_full':
        vgg = VggWithLCL(config['num_classes'], learning_rate=3e-4, dropout=0.2)
        vgg.load('models/vgg_with_lcl/VGG19_2022-04-04_183636__it13750_e2.pt')
        model = VggFull(vgg, learning_rate=config['learning_rate'], run_identifier="")
        del vgg
        if model_path is not None:
            model.load(model_path)
        return model

    elif model_key == 'tiny_cnn':
        model = TinyCNN(conv_channels=config['conv_channels'], num_classes=config['num_classes'], learning_rate=config['learning_rate'], run_identifier=config['run_identifier'])
        if model_path is not None:
            model.load(model_path)
        return model

    elif model_key == 'tiny_lateral_net':
        model = TinyLateralNet(conv_channels=config['conv_channels'], num_classes=config['num_classes'],
            learning_rate=config['learning_rate'], run_identifier=config['run_identifier'], num_multiplex=config['num_multiplex'],
            lcl_distance=config['lcl_distance'], lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        if model_path is not None:
            model.load(model_path)
        return model

    elif model_key == 'vgg16_lcl':
        model = SmallVggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
            num_multiplex=config['num_multiplex'], do_wandb=False, run_identifier="",
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        if model_path is not None:
            model.load(model_path)
        if config['use_lcl']:
            model.features.lcl3.enable()
        return model

    elif model_key == 'lcl':
        model = VggWithLCL(config['num_classes'], learning_rate=config['learning_rate'], dropout=config['dropout'],
            num_multiplex=config['num_multiplex'], do_wandb=False, run_identifier="",
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'])
        if model_path is not None:
            model.load(model_path)
        if config['use_lcl']:
            model.features.lcl3.enable()
        return model

    elif model_key in ['vgg19r_lcl', 'vgg19r_lcl__scaling', 'vgg19r_lcl__random_k_change', 'vgg19r_lcl__random_multiplex_selection', 'vgg19r_lcl__fc_only']:
        vgg = VggWithLCL(config['num_classes'], learning_rate=3e-4, dropout=0.2)
        vgg.load('models/vgg_with_lcl/VGG19_2022-04-04_183636__it13750_e2.pt')

        model = VGGReconstructionLCL(vgg, learning_rate=config['learning_rate'], after_pooling=config['after_pooling'],
            num_multiplex=config['num_multiplex'], run_identifier='', lcl_distance=config['lcl_distance'],
            lcl_alpha=config['lcl_alpha'], lcl_eta=config['lcl_eta'], lcl_theta=config['lcl_theta'], lcl_iota=config['lcl_iota'],
            use_scaling=config['use_scaling'],
            random_k_change=config['random_k_change'],
            random_multiplex_selection=config['random_multiplex_selection'],
            gradient_learn_k=config['gradient_learn_k'],
            fc_only=config['fc_only'])
        if model_path is not None:
            model.load(model_path)
        if not config['fc_only']:
            model.features.lcl.enable()
        del vgg
        return model
