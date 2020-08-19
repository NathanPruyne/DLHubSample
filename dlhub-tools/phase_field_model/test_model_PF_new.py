from dlhub_sdk.client import DLHubClient
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def apply_model(path):

    files = glob.glob(os.path.join(path, '*.png'))

    dl = DLHubClient()

    imgs = []
    for img_path in files:
        with open(img_path, 'rb') as fp:
            imgs = [np.array(Image.open(fp).convert('RGB'))]

        outs = dl.run('npruyne_globusid/dendrite_pf_segnet', imgs)

        print("Saving " + str(img_path.split('/')[-1]))
        plt.imsave('dlhub_results/PF_segnet/' + str(img_path.split('/')[-1]), outs[0], cmap='gray')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply images to the Phase Field model via DLHub")

    parser.add_argument("filepath", help="Path to a directory with files to process")

    args = parser.parse_args()

    apply_model(args.filepath)