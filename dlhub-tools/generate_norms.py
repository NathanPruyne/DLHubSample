import argparse
import os

from apply import average
from config import get_config

def generate_norms(arguments):
    #Obtains the configuration hyperparameters given the version and training dataset. This is solely
    # for obtaining the correct trained network in the segmentation process.
    config = get_config(arguments.dataset, arguments.version)
    
    avg, std = average(os.path.join(config['root'], 'train/images'))

    if not os.path.isdir('dlhub_norms'):
        os.mkdir('dlhub_norms')

    fullpath = arguments.output_name + '.txt'

    with open(fullpath, 'w') as f:
        f.write(str(avg) + '\n')
        f.write(str(std))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate norms file for use in DLHub upload')

    # Assigns the first two positional arguments to the dataset on which the network was trained
    # and the configuration version used which uniquely specify the trained network to apply.

    parser.add_argument('dataset', help='Dataset on which Network Trained')
    parser.add_argument('version', help='Version Defined in config.py: [v1, v2, ... ]')

    parser.add_argument('output_name', help='File name to output norms to')

    # Stores the command line arguments as a class with attributes corresponding to each argument,
    # then calls the application function on this class.
    args = parser.parse_args()
    generate_norms(args)