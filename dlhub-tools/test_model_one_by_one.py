from dlhub_sdk.client import DLHubClient
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def apply_model(path, model, outpath):

    files = glob.glob(os.path.join(path, '*.png'))

    dl = DLHubClient()

    imgs = []
    for img_path in files:
        with open(img_path, 'rb') as fp:
            imgs = [np.array(Image.open(fp).convert('RGB'))]

        outs = dl.run(model, imgs)

        print("Saving " + os.path.split(files[num])[1])
        plt.imsave(outpath + os.path.split(files[num])[1], seg, cmap='gray')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply images to a model via DLHub one image at a time")

    parser.add_argument("filepath", help="Path to a directory with files to process")
    parser.add_argument("model", help="DLHub model to apply the images to")
    parser.add_argument("outpath", help="Path to a directory to store result files in")

    args = parser.parse_args()

    apply_model(args.filepath, args.model, args.outpath)