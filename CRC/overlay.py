from PIL import Image
import numpy as np

def overlay(originaly, generatedy):
    originaly = originaly.convert('RGBA')
    generatedy.putalpha(70)
    for row in range(generatedy.size[0]):
        for col in range(generatedy.size[1]):
            if generatedy.getpixel((row, col)) == (0, 0, 0, 70):
                generatedy.putpixel((row, col), (0, 0, 0, 0))
    org_width, org_height = originaly.size
    generatedy = generatedy.resize((org_width, org_width), Image.BICUBIC)
    gen_width, gen_height = generatedy.size
    left = (gen_width - org_width) / 2
    top = (gen_height - org_height) / 2
    right = (gen_width + org_width) / 2
    bottom = (gen_height + org_height) / 2
    generatedy = generatedy.crop((left, top, right, bottom))
    overlay_img = Image.alpha_composite(originaly, generatedy)
    return overlay_img
