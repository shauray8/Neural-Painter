import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from utils import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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

cnn = models.vgg16(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


content_layers_default = ["conv_4"]
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

input_img = content_img.clone()

#plt.figure()
#imshow(input_img, title='Input Image')

num_steps = 300

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):

    print("Building the style transfer model ...")
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)


    print("Optimizing ...")

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
    
        
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

def make():
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps)

    plt.figure()
    imshow(output, title='Output Image')

    plt.ioff()
    plt.show()



size()
show_image()
