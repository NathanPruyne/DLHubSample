import torch

import argparse
import os
import sys

from config import get_config
from models import model_mappings

def generate_model(arguments):
    #Obtains the configuration hyperparameters given the version and training dataset. This is solely
    # for obtaining the correct trained network in the segmentation process.
    config = get_config(arguments.dataset, arguments.version)
    method = config['model']
    model_dir = './saved/%s_%s.pth' % (config['name'], method)
    try:
        model = model_mappings[method](K=config['n_class'])
        model.load_state_dict(torch.load(model_dir, map_location='cpu')['model_state_dict'], strict=False)
    except KeyError:
        print('%s model does not exist' % method)
        sys.exit(1)

    if not os.path.isdir('dlhub_models'):
        os.mkdir('dlhub_models')
    
    fullpath = arguments.output_name + '.pth'

    torch.save(model, fullpath)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate full model for DLHub upload from saved data')

    # Assigns the first two positional arguments to the dataset on which the network was trained
    # and the configuration version used which uniquely specify the trained network to apply.

    parser.add_argument('dataset', help='Dataset on which Network Trained')
    parser.add_argument('version', help='Version Defined in config.py: [v1, v2, ... ]')

    parser.add_argument('output_name', help='File name to output full model to')

    # Stores the command line arguments as a class with attributes corresponding to each argument,
    # then calls the application function on this class.
    args = parser.parse_args()
    generate_model(args)
