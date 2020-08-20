from dlhub_sdk.client import DLHubClient
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data_path = './data/tomography/test/images'
files = glob.glob(os.path.join(data_path, '*.png'))

imgs = []
for img_path in files:
    with open(img_path, 'rb') as fp:
        img = Image.open(fp).convert('RGB')

        imgs.append(np.array(Image.open(fp).convert('RGB')))

dl = DLHubClient()

dl._fx_client.max_request_size = 50 * 1024 ** 3

outs = dl.run('npruyne_globusid/dendrite_pf_segnet', imgs)

num = 0

for seg in outs:

    print("Saving " + str(files[num].split('/')[-1]))
    plt.imsave('dlhub_results/PF_segnet/' + str(files[num].split('/')[-1]), seg, cmap='gray')
    num += 1