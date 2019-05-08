from torch.utils.data.dataset import Dataset
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
import random

class crc_Dataset(Dataset):
    def __init__(self, img_size, job):
        self.transforms = transforms.Compose([
            transforms.Scale(size=img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()])
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
        label_as_img = Image.open(single_label_name).convert('RGB')
        img_tensor = self.transforms(img_as_img)
        label_tensor = self.transforms(label_as_img)
        return (img_tensor, label_tensor)

    def __len__(self):
        return self.data_len