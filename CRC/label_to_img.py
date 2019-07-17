from PIL import Image
import numpy

def label_to_img(batch, img_size): #[4, 11, 128, 128]
    height = img_size
    width = img_size
    element = batch[0].numpy()
    out_img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            max_channel = 0
            max_value = element[0, row, col]
            for channel in range(11):
                if element[channel, row, col] > max_value:
                    max_channel = channel
                    max_value = element[channel, row, col]
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
            elif max_channel == 6 :
                pixel = (187, 155, 25, 255)
            elif max_channel == 7 :
                pixel = (0, 255, 125, 255)
            elif max_channel == 8 :
                pixel = (255, 255, 125, 255)
            elif max_channel == 9 :
                pixel = (123, 15, 175, 255)
            elif max_channel == 10 :
                pixel = (124, 155, 5, 255)
            out_img.putpixel((col, row), pixel)
    return out_img