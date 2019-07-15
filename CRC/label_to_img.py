from PIL import Image
import numpy

def label_to_img(pred_list, img_size): #[4, 11, 128, 128]
    height = img_size
    width = img_size

    minimum = 0.55

    element_1 = (pred_list[0][0].cpu().data).numpy()
    element_2 = (pred_list[1][0].cpu().data).numpy()
    element_3 = (pred_list[2][0].cpu().data).numpy()
    element_4 = (pred_list[3][0].cpu().data).numpy()
    element_5 = (pred_list[4][0].cpu().data).numpy()
    element_6 = (pred_list[5][0].cpu().data).numpy()

    #backgound in allen head
    out_img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            max_channel = 0
            max_value = element_1[0, row, col]
            for channel in range(1,6):
                if element_1[channel, row, col] > element_1[max_channel, row, col]:
                    max_channel = channel
                    max_value = element_1[channel, row, col]
            if max_channel == 0 :
                pixel =  (0, 0, 0, 255)
                max_value = minimum  #so werden die anderen heads bevorzugt
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

            if 0.5 < element_2[1, row, col] and (element_2[1, row, col]-0.5)/0.5 > (max_value-0.2)/0.8:
                pixel = (187, 155, 25, 255)
                max_value = element_2[1, row, col]
            if 0.5 < element_3[1, row, col] and (element_3[1, row, col]-0.5)/0.5 > (max_value-0.2)/0.8:
                pixel = (0, 255, 125, 255)
                max_value = element_3[1, row, col]
            if 0.5 < element_4[1, row, col] and (element_4[1, row, col]-0.5)/0.5 > (max_value-0.2)/0.8:
                pixel = (255, 255, 125, 255)
                max_value = element_4[1, row, col]
            if 0.5 < element_5[1, row, col] and (element_5[1, row, col]-0.5)/0.5 > (max_value-0.2)/0.8:
                pixel = (123, 15, 175, 255)
                max_value = element_5[1, row, col]
            if 0.5 < element_6[1, row, col] and (element_6[1, row, col]-0.5)/0.5 > (max_value-0.2)/0.8:
                pixel = (124, 155, 5, 255)
                max_value = element_6[1, row, col]
            out_img.putpixel((col, row), pixel)

    #einzelbilder
    out_img_1 = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            max_channel = 0
            for channel in range(6):
                if element_1[channel, row, col] > element_1[max_channel, row, col]:
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
            out_img_1.putpixel((col, row), pixel)

    out_img_2 = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            if element_2[0, row, col] < element_2[1, row, col]  and element_2[1, row, col] > minimum:
                pixel = (187, 155, 25, 255)
                out_img_2.putpixel((col, row), pixel)

    out_img_3 = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            if element_3[0, row, col] < element_3[1, row, col]  and element_3[1, row, col] > minimum:
                pixel = (0, 255, 125, 255)
                out_img_3.putpixel((col, row), pixel)

    out_img_4 = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            if element_4[0, row, col] < element_4[1, row, col] and element_4[1, row, col] > minimum:
                pixel = (255, 255, 125, 255)
                out_img_4.putpixel((col, row), pixel)

    out_img_5 = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            if element_5[0, row, col] < element_5[1, row, col]  and element_5[1, row, col] > minimum:
                pixel = (123, 15, 175, 255)
                out_img_5.putpixel((col, row), pixel)

    out_img_6 = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            if element_6[0, row, col] < element_6[1, row, col] and element_6[1, row, col] > minimum:
                pixel = (124, 155, 5, 255)
                out_img_6.putpixel((col, row), pixel)

    return out_img, out_img_1, out_img_2, out_img_3, out_img_4, out_img_5, out_img_6