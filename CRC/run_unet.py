from UNet import *
from crc_dataset import *
from label_to_img import *
from overlay import *
from dice import *
import sys
import torch


use_gpu = torch.cuda.is_available()

def main():
    img_size = 128
    epoch = 5000
    batch_size = 12
    learning_rate = 0.1
    momentum = 0.9

    job = sys.argv[1]
    model = sys.argv[2]

    if job == 'train':
        train_data = crc_Dataset(img_size=img_size, job=job)
        train_loader = data.DataLoader(dataset = train_data, batch_size = batch_size, pin_memory=True, shuffle=True)

        try:
            generator = torch.load('model/'+ model)
            try:
                learning_rate = float(sys.argv[3])
            except:
                pass
        except:
            generator = UnetGenerator(3, 11, 64).cuda()# (3,3,64)#in_dim,out_dim,num_filter out dim = 4 oder 11
            print("new model generated")
        loss_function = nn.MSELoss()
        #loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
        #optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True)

        for ep in range(epoch):
            dice_sum = 0
            for batch_number,(input_batch, label_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                input_batch = Variable(input_batch).cuda(0)
                label_batch = Variable(label_batch).cuda(0)
                generated_batch = generator.forward(input_batch)
                loss = loss_function(generated_batch, label_batch)
                dice = dice_loss(generated_batch, label_batch.cuda()).item()
                dice_sum += dice
                loss.backward()
                optimizer.step()
            avg_dice = dice_sum/train_loader.__len__()
            print("epoche:{}/{} avg dice:{}".format(ep, epoch - 1, avg_dice))
            scheduler.step(avg_dice)
            if ep % 10 == 0:
                torch.save(generator, 'model/'+model)
                print("model saved")
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
        original_list = []
        for ordner in listdir('data/validate'):
            files = listdir('data/validate/' + ordner + '/left_frames')
            for bild in files:
                if bild.find('.png') != -1:
                    original_list.append('data/validate/' + ordner + '/left_frames/' + bild)
        dice_sum = 0
        for batch_number, (input_batch, label_batch) in enumerate(validate_loader):
            original = Image.open(original_list.pop(0))
            input_batch = Variable(input_batch).cuda(0)
            generated_batch= generator.forward(input_batch)
            dice = dice_loss(generated_batch, label_batch.cuda()).item()
            dice_sum += dice
            print("batch:{}/{} dice: {}".format(batch_number, validate_loader.__len__()-1, dice))
            generated_out_img = label_to_img(generated_batch.cpu().data, img_size)
            label_out_img = label_to_img(label_batch.cpu().data, img_size)
            overlay_img = overlay(original.copy(), generated_out_img.copy())
            #generated_out_img.save("data/validate-result/img_{}_generated.png".format(batch_number))
            #label_out_img.save("data/validate-result/img_{}_truth.png".format(batch_number))
            overlay_img.save("data/validate-result/img_{}_original.png".format(batch_number))
        avg_dice = dice_sum / validate_loader.__len__()
        print("Avgerage dice distance, 0 means perfect:", avg_dice)

if __name__ == "__main__":
    main()
    pass