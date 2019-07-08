from PIL import Image
import numpy

def label_to_img(pred_list, img_size): #[4, 11, 128, 128]
    height = img_size
    width = img_size
    out_img = Image.new('RGBA', (width, height), (0, 0, 0, 255))

    element_1 = (pred_list[0][0].cpu().data).numpy()
    element_2 = (pred_list[1][0].cpu().data).numpy()
    element_3 = (pred_list[2][0].cpu().data).numpy()
    element_4 = (pred_list[3][0].cpu().data).numpy()
    element_5 = (pred_list[4][0].cpu().data).numpy()
    element_6 = (pred_list[5][0].cpu().data).numpy()
    for row in range(height):
        for col in range(width):
            max_channel = 0
            max_value = element_1[0, row, col]
            for channel in range(5):
                if element_1[channel, row, col] > element_1[max_channel, row, col]:
                    max_channel = channel
                    max_value = element_1[channel, row, col]
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

            if element_2[6, row, col] > max_value:
                pixel = (187, 155, 25, 255)
                max_value = element_2[6, row, col]
            if element_3[7, row, col] > max_value:
                pixel = (0, 255, 125, 255)
                max_value = element_2[7, row, col]
            if element_4[8, row, col] > max_value:
                pixel = (255, 255, 125, 255)
                max_value = element_2[8, row, col]
            if element_5[9, row, col] > max_value:
                pixel = (123, 15, 175, 255)
                max_value = element_2[9, row, col]
            if element_6[10, row, col] > max_value:
                pixel = (124, 155, 5, 255)
                max_value = element_2[10, row, col]
            out_img.putpixel((col, row), pixel)
    return out_img