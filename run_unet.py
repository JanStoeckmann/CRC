from UNet import *
from crc_dataset import *
from label_to_img import *
from overlay import *
from dice import *
import sys
import torch


use_gpu = torch.cuda.is_available()

def main():
    img_size = 256
    epoch = 5000
    batch_size = 10
    learning_rate = 0.01
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True)
        for ep in range(epoch):
            dice_sum = 0
            for batch_number,(input_batch, label_1_batch, label_2_batch, label_3_batch, label_4_batch, label_5_batch, label_6_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                input_batch = Variable(input_batch).cuda(0)
                label_1_batch = Variable(label_1_batch).cuda(0)
                label_2_batch = Variable(label_2_batch).cuda(0)
                label_3_batch = Variable(label_3_batch).cuda(0)
                label_4_batch = Variable(label_4_batch).cuda(0)
                label_5_batch = Variable(label_5_batch).cuda(0)
                label_6_batch = Variable(label_6_batch).cuda(0)
                generated_1_batch, generated_2_batch, generated_3_batch, generated_4_batch, generated_5_batch, generated_6_batch = generator.forward(input_batch)
                pred_target = [[generated_1_batch, label_1_batch], [generated_2_batch, label_2_batch], [generated_3_batch, label_3_batch], [generated_4_batch, label_4_batch], [generated_5_batch, label_5_batch], [generated_6_batch, label_6_batch]]
                loss_1 = loss_function(generated_1_batch, label_1_batch)
                loss_2 = loss_function(generated_2_batch, label_2_batch)
                loss_3 = loss_function(generated_3_batch, label_3_batch)
                loss_4 = loss_function(generated_4_batch, label_4_batch)
                loss_5 = loss_function(generated_5_batch, label_5_batch)
                loss_6 = loss_function(generated_6_batch, label_6_batch)
                loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
                total = 0.
                for [pred,target] in pred_target:
                    total += dice_loss(pred,target.cuda()).item()
                dice = total/6
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
        for batch_number, (input_batch, label_1_batch, label_2_batch, label_3_batch, label_4_batch, label_5_batch, label_6_batch) in enumerate(validate_loader):
            original = Image.open(original_list.pop(0))
            input_batch = Variable(input_batch).cuda(0)
            label_1_batch = Variable(label_1_batch).cuda(0)
            label_2_batch = Variable(label_2_batch).cuda(0)
            label_3_batch = Variable(label_3_batch).cuda(0)
            label_4_batch = Variable(label_4_batch).cuda(0)
            label_5_batch = Variable(label_5_batch).cuda(0)
            label_6_batch = Variable(label_6_batch).cuda(0)
            generated_1_batch, generated_2_batch, generated_3_batch, generated_4_batch, generated_5_batch, generated_6_batch = generator.forward(
                input_batch)
            pred_target = [[generated_1_batch, label_1_batch], [generated_2_batch, label_2_batch],
                           [generated_3_batch, label_3_batch], [generated_4_batch, label_4_batch],
                           [generated_5_batch, label_5_batch], [generated_6_batch, label_6_batch]]
            dice = multiple_dice(pred_target)
            dice_sum += dice
            print("batch:{}/{} dice: {}".format(batch_number, validate_loader.__len__()-1, dice))
            pred_list = generated_1_batch, generated_2_batch, generated_3_batch, generated_4_batch, generated_5_batch, generated_6_batch
            generated_out_img = label_to_img(pred_list, img_size)
            overlay_img = overlay(original.copy(), generated_out_img.copy())
            overlay_img.save("data/validate-result/img_{}_original.png".format(batch_number))
        avg_dice = dice_sum / validate_loader.__len__()
        print("Avgerage dice distance, 0 means perfect:", avg_dice)

if __name__ == "__main__":
    main()
    pass