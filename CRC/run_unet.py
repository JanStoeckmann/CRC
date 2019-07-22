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
    batch_size = 14
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
                dice = dice_loss(generated_batch.cpu(),label_batch.cpu())
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

        each_dice = np.zeros((11, 2))#channel[0-10], [dice_sum, c1]

        for batch_number, (input_batch, label_batch) in enumerate(validate_loader):
            origi = original_list.pop(0)
            print(origi)
            original = Image.open(origi)
            input_batch = Variable(input_batch).cuda(0)
            generated_batch= generator.forward(input_batch)
            dice = dice_loss(generated_batch.cpu(),label_batch.cpu())
            dice_sum += dice
            for chan in range(1,11):
                di = dice_each(generated_batch.cpu(), label_batch.cpu(), chan)
                if type(di) is float:
                    each_dice[chan][0] += di
                    each_dice[chan][1] += 1

            print("img:{}/{} dice: {}".format(batch_number, validate_loader.__len__()-1, dice))
            generated_out_img = label_to_img(generated_batch.cpu().data, img_size)
            overlay_img = overlay(original.copy(), generated_out_img.copy())
            overlay_img.save("data/validate-result/img_{}_overlay.png".format(batch_number))
            generated_out_img.save("data/validate-result/img_{}_generated.png".format(batch_number))
            original.save("data/validate-result/img_{}_original.png".format(batch_number))
        print("\nErgebnis:\n")
        for chan in range(1, 11):
            if each_dice[chan][1] != 0:
                avg = each_dice[chan][0] / each_dice[chan][1]
                print("Dice Klasse", chan, ":", avg)
            else:
                print("Dice Klasse", chan, ": Klasse nicht vertreten")
        avg_dice = dice_sum / validate_loader.__len__()
        print("Dice alle Klassen :", avg_dice)

if __name__ == "__main__":
    main()
    pass