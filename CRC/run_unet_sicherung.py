from UNet import *
from crc_dataset import *
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
import random
import time
import numpy


use_gpu = torch.cuda.is_available()

def rgbaToChannel(label_img):
    label_channel = []
    channel_row = []
    pixel_list = list(label_img.getdata())
    for row in range(0,1024):
        col = 0
        for col in range(0,1280):
            channel_value = ()  #eckige oder runde klammer?
            pixel_value = pixel_list[row*1280+col]
            if pixel_value == (0, 0, 0, 255):
                channel_value = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            elif pixel_value == (0, 255, 0, 255):
                channel_value = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            elif pixel_value == (0, 255, 255, 255):
                channel_value = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
            elif pixel_value == (125, 255, 12, 255):
                channel_value = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
            elif pixel_value == (255, 55, 0, 255):
                channel_value = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
            elif pixel_value == (24, 55, 125, 255):
                channel_value = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
            elif pixel_value == (187, 155, 25, 255):
                channel_value = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
            elif pixel_value == (0, 255, 125, 255):
                channel_value = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            elif pixel_value == (255, 255, 125, 255):
                channel_value = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
            elif pixel_value == (123, 15, 175, 255):
                channel_value = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
            elif pixel_value == (124, 155, 5, 255):
                channel_value = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
            else:
                channel_value = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            channel_row.append(channel_value)
        label_channel.append(channel_row)
        channel_row = []
    return label_channel

def main():
    img_size = 128
    epoch = 20
    batch_size = 8
    learning_rate = 0.01
    momentum = 0.99

    if sys.argv[1] == 'train-new' or sys.argv[1] == 'train-model':
        train_data = crc_Dataset(img_size=img_size, job="train")
        train_loader = data.DataLoader(dataset = train_data, batch_size = batch_size, pin_memory=True, shuffle=True) #shuffle=True

        if sys.argv[1] == 'train-model':
            generator = torch.load('model/model1.pkl')
        else:
            generator = UnetGenerator(3, 3, 64).cuda()# (3,3,64)#in_dim,out_dim,num_filter out dim = 4 oder 11
        recon_loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
        #optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

        for ep in range(epoch):
            for batch_number,(input_batch, label_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                input_batch = Variable(input_batch).cuda(0)
                label_batch = Variable(label_batch).cuda(0)
                generated_batch = generator.forward(input_batch)
                loss = recon_loss_func(generated_batch, label_batch)
                print("epoche:{}/{} batch:{}/{} loss:{}".format(ep,epoch-1,batch_number,train_loader.__len__()-1,loss.item()))
                loss.backward()
                optimizer.step()
                if ep % 2 == 0 and batch_number==0:
                    v_utils.save_image(generated_batch.cpu().data, "result/gen_image_{}_{}.png".format(ep, batch_number))
                if ep % 5 == 0 and batch_number == 0:
                    torch.save(generator, 'model/model1.pkl')
        torch.save(generator, 'model/model1.pkl')

    if sys.argv[1] == 'validate':
        validate_data = crc_Dataset(img_size=img_size, job="train")
        validate_loader = data.DataLoader(dataset=validate_data, batch_size=batch_size, pin_memory=True, shuffle=True)  # shuffle=True
        generator = torch.load('model/model1.pkl')
        recon_loss_func = nn.MSELoss()
        for batch_number, (input_batch, label_batch) in enumerate(validate_loader):
            input_batch = Variable(input_batch).cuda(0)
            label_batch = Variable(label_batch).cuda(0)
            generated_batch= generator.forward(input_batch)
            loss = recon_loss_func(generated_batch, label_batch)
            print("batch:{}/{} loss:{}".format(batch_number, validate_loader.__len__()-1, loss.item()))
            v_utils.save_image(generated_batch.cpu().data, "validate/gen_image_{}.png".format(batch_number))

if __name__ == "__main__":
    main()
    pass