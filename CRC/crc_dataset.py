from torch.utils.data.dataset import Dataset
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
import random
import numpy as np

class crc_Dataset(Dataset):
    def __init__(self, img_size, job):
        self.img_size = img_size
        self.job = job
        self.img_transforms = transforms.Compose([
            transforms.Resize(size=img_size),#scallieren
            transforms.CenterCrop(img_size),#rand abschneiden
            transforms.ToTensor()])
        self.label_transforms = transforms.ToTensor()
        job = job + "/"
        self.image_list = []
        self.label_list = []
        for ordner in listdir('data/' + job):
            files = listdir('data/' + job + ordner + '/left_frames')
            for bild in files:
                if bild.find('.png') != -1:
                    self.image_list.append('data/' + job + ordner + '/left_frames/' + bild)
                    self.label_list.append('data/' + job + ordner + '/labels/' + bild)
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        single_image_name = self.image_list[index]
        single_label_name = self.label_list[index]
        img_as_img = Image.open(single_image_name)
        label_as_img = Image.open(single_label_name)
        if self.job == "train":
            if random.choice([True, False]):
                img_as_img = img_as_img.transpose(Image.FLIP_LEFT_RIGHT)
                label_as_img = label_as_img.transpose(Image.FLIP_LEFT_RIGHT)
            degree = random.randint(0, 359)
            img_as_img = img_as_img.rotate(degree, expand=True)
            label_as_img = label_as_img.rotate(degree, expand=True)
            wi, hei = img_as_img.size
            wizoom = random.randint(0, int(0.55 * wi))
            heizoom = int(wizoom/wi*hei)
            newwi = wi - wizoom
            newhei = hei - heizoom
            left = (wi - newwi) / 2
            top = (hei - newhei) / 2
            right = (wi + newwi) / 2
            bottom = (hei + newhei) / 2
            img_as_img = img_as_img.crop((left, top, right, bottom))
            label_as_img = label_as_img.crop((left, top, right, bottom))

        img_tensor = self.img_transforms(img_as_img)
        width, height = label_as_img.size
        new_height = self.img_size
        scale = (new_height/ float(height))
        new_width = int((float(width) * float(scale)))
        label_as_img = label_as_img.resize((new_width, new_height))
        width, height = label_as_img.size
        left = (width - self.img_size) / 2
        top = (height - new_height) / 2
        right = (width + self.img_size) / 2
        bottom = (height + new_height) / 2
        label_as_img = label_as_img.crop((left, top, right, bottom))
        width, height = label_as_img.size
        label_channel = np.zeros((height, width, 11), np.float32)
        for row in range(height):
            for col in range(width):
                pixel_value = label_as_img.getpixel((col, row))
                channel = -1
                if pixel_value == (0, 0, 0, 255):
                    channel = 0
                elif pixel_value == (0, 255, 0, 255):
                    channel = 1
                elif pixel_value == (0, 255, 255, 255):
                    channel = 2
                elif pixel_value == (125, 255, 12, 255):
                    channel = 3
                elif pixel_value == (255, 55, 0, 255):
                    channel = 4
                elif pixel_value == (24, 55, 125, 255):
                    channel = 5
                elif pixel_value == (187, 155, 25, 255):
                    channel = 6
                elif pixel_value == (0, 255, 125, 255):
                    channel = 7
                elif pixel_value == (255, 255, 125, 255):
                    channel = 8
                elif pixel_value == (123, 15, 175, 255):
                    channel = 9
                elif pixel_value == (124, 155, 5, 255):
                    channel = 10
                if channel != -1:
                    label_channel[row, col, channel] = 1

        label_tensor = self.label_transforms(label_channel)
        return (img_tensor, label_tensor)

    def __len__(self):
        return self.data_len