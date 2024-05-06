import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2 as cv
import numpy as np

import os

def import_image(image_path):
    image = cv.imread(image_path)
    return image

def find_images(image_path):
    images = []
    os.chdir(image_path)
    for image in os.listdir(image_path):
        images.append(image)
    
    image = import_image(image_path)
    return image