from UNet import *
from crc_dataset import *
from label_to_img import *
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
import random
import time
import numpy

def label_to_img(batch, img_size, batch_size): #[4, 11, 128, 128]
    height = img_size
    width = img_size
    element = batch[0].numpy()
    out_img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            execute=True
            max_channel = 0
            for channel in range(11):
                if element[channel, row, col] > element[max_channel, row, col]:
                    max_channel = channel
            if max_channel == 0 :
                pixel =  (0, 0, 0, 255)
            elif max_channel == 1 :
                pixel = (0, 255, 0, 255)
            elif max_channel == 2 :
                pixel = (0, 255, 255, 255)
            elif max_channel == 3 :
                pixel = (125, 255, 12, 255)
            elif max_channel == 4 :
                pixel = (255, 55, 0, 255)
            elif max_channel == 5 :
                pixel = (24, 55, 125, 255)
            elif max_channel == 6 :
                pixel = (187, 155, 25, 255)
            elif max_channel == 7 :
                pixel = (0, 255, 125, 255)
            elif max_channel == 8 :
                pixel = (255, 255, 125, 255)
            elif max_channel == 9 :
                pixel = (123, 15, 175, 255)
            elif max_channel == 10 :
                pixel = (124, 155, 5, 255)
            out_img.putpixel((col, row), pixel)
    return out_img