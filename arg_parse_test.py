import argparse
import yaml

# def load_config(config_path):
    # with open(config_path, 'r') as config_file:
        # config = yaml.safe_load(config_file)
    # return config

# config = load_config('./asymmetric_conf.yaml')

with open('./asymmetric_conf.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

batch_sizes = config['batch_sizes']
learning_rates = config['learning_rates']
coefficients = config['coefficients']
n_epochs = config['n_epochs']

print("Batch sizes:", batch_sizes)
print("Learning rates:", learning_rates['backbone_lr'])
print("Coefficients:", coefficients)
print("Number of epochs:", n_epochs)