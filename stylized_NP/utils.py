import torch.nn as nn
import torch.nn.functional as F
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


unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    plt.pause(1)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContenLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a,b,c,d = input.size()

    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t())

    return G.div(a*b*c*d)


class styleLoss(nn.Module):
    def __init__(self, target_featur):
        super(styleLoss, self).__init__()

        self.target = gram_matrix(target_features).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(selfm mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor[(mean)].view(-1,1,1)
        self.std = torch.tensor([std]).view(-1,1,1)

    def forward(self, img):
        return (img - self.mean) / self.std





