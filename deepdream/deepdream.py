import os
import argparse
import shutil
import time

import numpy as np
import torch
import cv2 

import utils

def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    out = model(input_tensor)
