from torch.utils.data.dataset import Dataset
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
import random
import numpy as np

class crc_validate_Dataset(Dataset):
    def __init__(self, img_size, job):
        self.img_size = img_size
        self.img_transforms = transforms.Compose([
            transforms.Resize(size=img_size),#scallieren
            transforms.CenterCrop(img_size),#rand abschneiden
            transforms.ToTensor()])
        job = job + "/"
        self.image_list = []
        for ordner in listdir('data/' + job):
            files = listdir('data/' + job + ordner + '/left_frames')
            for bild in files:
                if bild.find('.png') != -1:
                    self.image_list.append('data/' + job + ordner + '/left_frames/' + bild)
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        single_image_name = self.image_list[index]
        img_as_img = Image.open(single_image_name)
        img_tensor = self.img_transforms(img_as_img)
        return img_tensor

    def __len__(self):
        return self.data_len