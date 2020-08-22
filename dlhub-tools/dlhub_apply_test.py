from dlhub_apply import apply

from dlhub_sdk.client import DLHubClient
import glob
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_model(model_file, norms_file, path, outpath):
    data_path = path
    files = glob.glob(os.path.join(data_path, '*.png'))

    imgs = []
    for img_path in files:
        with open(img_path, 'rb') as fp:
            imgs.append(np.array(Image.open(fp).convert('RGB')))

    outs = apply(imgs, model_file=model_file, norms_file=norms_file)

    num = 0

    for seg in outs:

        print("Saving " + os.path.split(files[num])[1])
        plt.imsave(outpath + os.path.split(files[num])[1], seg, cmap='gray')
        num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply images to a model via DLHub")

    parser.add_argument('model_file', help='.pth file containing model information')
    parser.add_argument('norms_file', help='.txt file containing norms information')
    parser.add_argument("filepath", help="Path to a directory with files to process")
    parser.add_argument("outpath", help="Path to a directory to store result files in")

    args = parser.parse_args()

    apply_model(args.model_file, args.norms_file, args.filepath, args.outpath)