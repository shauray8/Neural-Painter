import torch 
import torchvision.models as models
import torch.nn as nn

vgg16 = models.vgg16(pretrained = True)
print(vgg16)

for params in vgg16.parameters():
    params.requires_grad = False

vgg16.conv = nn.Conv2d(1000, 3, 4, padding=1)

print(vgg16)

path = "../../data/Dog/0.png"
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(path)

