import torch.nn as nn
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def loader(imsize):
    loader = transforms.compose([
        transforms.Resize=(imsize),
        transforms.ToTensor()])
    return loader

def img_loader(image, device):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)



