import os
import argparse

def run_experiment(config):
    cmd = ' '.join([
        'python run_lateral_model.py --lcl',
        '--lr ' + str(config['learning_rate']),
        '--dropout ' + str(config['dropout']),
        '--num_epochs ' + str(config['num_epochs']),
        '--num_multiplex ' + str(config['num_multiplex']),
        '--batch_size ' + str(config['batch_size']),
        '--lcl_alpha ' + str(config['lcl_alpha']),
        '--lcl_theta ' + str(config['lcl_theta']),
        '--lcl_eta ' + str(config['lcl_eta']),
        '--lcl_iota ' + str(config['lcl_iota']),
    ])
    os.system(cmd)

def learning_rate_exploration(args, config):
    for lr in [1e-3, 1e-4, 3e-5, 1e-5]:
        config['learning_rate'] = lr
        run_experiment(config)

def alpha_exploration(args, config):
    for alpha in [1e-2, 3e-3, 3e-4, 1e-5]:
        config['lcl_alpha'] = alpha
        run_experiment(config)

def theta_exploration(args, config):
    for theta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        config['lcl_theta'] = theta
        run_experiment(config)

def eta_exploration(args, config):
    for eta in [0, 0.25, 0.5, 0.75, 1.0]:
        config['lcl_eta'] = eta
        run_experiment(config)

def iota_exploration(args, config):
    for iota in [0.1, 0.3, 0.5, 0.7, 0.9]:
        config['lcl_iota'] = iota
        run_experiment(config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=False, action='store_true', help='Learning rate exploration')
    parser.add_argument('--alpha', default=False, action='store_true', help='alpha exploration (K change)')
    parser.add_argument('--theta', default=False, action='store_true', help='theta exploration (output = (1-theta)*A+theta*L)')
    parser.add_argument('--eta', default=False, action='store_true', help='eta exploration (noise)')
    parser.add_argument('--iota', default=False, action='store_true', help='iota exploration (argmax[(1-iota)*A+iota*L])')

    args = parser.parse_args()

    base_config = {
        'learning_rate': 3e-4,
        'dropout': 0.2,
        'num_epochs': 4,
        'num_multiplex': 4,
        'batch_size': 10,
        'lcl_alpha': 1e-3,
        'lcl_theta': 0.2,
        'lcl_eta': 0.1,
        'lcl_iota': 0.5
    }

    if args.lr:
        learning_rate_exploration(args, base_config)

    if args.alpha:
        alpha_exploration(args, base_config)

    if args.theta:
        theta_exploration(args, base_config)

    if args.eta:
        eta_exploration(args, base_config)

    if args.iota:
        iota_exploration(args, base_config)

    print('Running just the base config.')
    for i in range(3):
        run_experiment(base_config)

if __name__ == '__main__':
    main()
