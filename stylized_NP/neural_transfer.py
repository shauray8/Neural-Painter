import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "../../data/style_transfer/image/"
style_img = img_loader("./test_img/Albrecht_Dürer_10.jpg", device)
content_img = img_loader("./test_img/El_Greco_1.jpg", device)

def size():
    print(style_img.shape)
    print(content_img.shape)


def show_image():
    plt.figure()
    imshow(style_img, title="style image")
    plt.figure()
    imshow(content_img, title="Content image")

show_image()
