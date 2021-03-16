import os
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        pass

    def forward(self, x):
        pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    D, G = Discriminator(), Generator()
    print(D,G)
