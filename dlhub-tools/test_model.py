from dlhub_sdk.client import DLHubClient
import glob
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_model(path, model, outpath):
    data_path = path
    files = glob.glob(os.path.join(data_path, '*.png'))

    imgs = []
    for img_path in files:
        with open(img_path, 'rb') as fp:
            imgs.append(np.array(Image.open(fp).convert('RGB')))

    dl = DLHubClient()

    dl._fx_client.max_request_size = 50 * 1024 ** 3

    outs = dl.run(model, imgs)

    num = 0

    for seg in outs:

        print("Saving " + str(files[num].split('/')[-1]))
        plt.imsave(outpath + str(files[num].split('/')[-1]), seg, cmap='gray')
        num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply images to a model via DLHub")

    parser.add_argument("filepath", help="Path to a directory with files to process")
    parser.add_argument("model", help="DLHub model to apply the images to")
    parser.add_argument("outpath", help="Path to a directory to store result files in")

    args = parser.parse_args()

    apply_model(args.filepath, args.model, args.outpath)