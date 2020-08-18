from dlhub_apply import apply

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def test_apply():

    data_path = './data/tomography/test/images/c70_ObjT1_z171.png'
    imgs = []
    with open(data_path, 'rb') as fp:
        imgs.append(np.array(Image.open(fp).convert('RGB')))
    out = apply(imgs, model_file='PFNetwork.pth', norms_file='PFnormsnew.txt')
    print(out[0].shape)

    for seg in out:
        plt.imsave('dlhub_result_new_norms.png', seg, cmap='gray')    

    return out

if __name__ == "__main__":
    out = test_apply()
    print(out)