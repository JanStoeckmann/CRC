from UNet import *
from crc_dataset import *
from label_to_img import *
from dice import *
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

def main():
    img_size = 128
    epoch = 2
    batch_size = 20 #40
    learning_rate = 0.01
    momentum = 0.99

    job = sys.argv[1]
    model = sys.argv[2]

    if job == 'train':
        train_data = crc_Dataset(img_size=img_size, job=job)
        train_loader = data.DataLoader(dataset = train_data, batch_size = batch_size, pin_memory=True, shuffle=True) #shuffle=True

        try:
            generator = torch.load('model/'+ model)
        except:
            generator = UnetGenerator(3, 11, 64).cuda()# (3,3,64)#in_dim,out_dim,num_filter out dim = 4 oder 11
        loss_function = nn.MSELoss()
        #loss_function = nn.L1Loss()
        optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
        #optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

        for ep in range(epoch):
            for batch_number,(input_batch, label_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                input_batch = Variable(input_batch).cuda(0)
                label_batch = Variable(label_batch).cuda(0)
                generated_batch = generator.forward(input_batch)
                loss = loss_function(generated_batch, label_batch)
                print("epoche:{}/{} batch:{}/{} loss:{}".format(ep,epoch-1,batch_number,train_loader.__len__()-1,loss.item()))
                loss.backward()
                optimizer.step()
                if ep % 1 == 0 and batch_number==0:
                    batch_out_img = label_to_img(generated_batch.cpu().data, img_size, batch_size)
                    batch_out_img.save("data/train-result/gen_{}_{}.png".format(ep, batch_number))
                if ep % 5 == 0 and batch_number == 0:
                    torch.save(generator, 'model/'+model)
        torch.save(generator, 'model/'+model)

    if job == 'validate':
        batch_size = 1
        validate_data = crc_Dataset(img_size=img_size, job=job)
        validate_loader = data.DataLoader(dataset=validate_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        try:
            generator = torch.load('model/'+model)
        except:
            print("Error: Model doesn't exist")
            exit()
        dice_sum = 0
        for batch_number, (input_batch, label_batch) in enumerate(validate_loader):
            input_batch = Variable(input_batch).cuda(0)
            generated_batch= generator.forward(input_batch)
            dice = dice_loss(generated_batch, label_batch.cuda()).item()
            dice_sum += abs(dice)
            print("batch:{}/{} dice: {}".format(batch_number, validate_loader.__len__()-1, dice))
            generated_out_img = label_to_img(generated_batch.cpu().data, img_size, batch_size)
            label_out_img = label_to_img(label_batch.cpu().data, img_size, batch_size)
            generated_out_img.save("data/validate-result/img_{}_generated.png".format(batch_number))
            label_out_img.save("data/validate-result/img_{}_truth.png".format(batch_number))
        avg_dice = dice_sum / validate_loader.__len__()
        print("Avgerage dice distance, 0 means perfect:", avg_dice)

if __name__ == "__main__":
    main()
    pass