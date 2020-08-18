import torch
import torchvision
import numpy as np

import ast

import models

def apply(input_files, model_file, norms_file):

    model = torch.load(model_file)
    model.eval()

    with open(norms_file, mode='r') as file:
        avg = ast.literal_eval(file.readline())
        std = ast.literal_eval(file.readline())
    
    transformer=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=avg, std=std)
    ])

    output_files = []

    for img in input_files:
        

        img = transformer(img)
        img = torch.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))

        outputs = model(img)

        predictions = outputs.argmax(1)

        #print(type(predictions[0]))

        output_files.append(np.array(predictions[0]))

    return output_files
