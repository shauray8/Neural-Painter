import torch.nn as nn
import torch
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

imsize = 512

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

def img_loader(image_name, device):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)



