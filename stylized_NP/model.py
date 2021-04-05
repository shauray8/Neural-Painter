import torch
import numpy 
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=True).features
        pass

    def forward(self, x):
        pass



if __name__ == "__main__":
    pass

