import os
import card_manipulation as cm

from PIL import Image
from skimage.io import imread

full_path = os.path.dirname(__file__)

pictures = os.listdir(full_path + 'data/cropped_angles/')
iterator = 0
for card in pictures:
    img = imread(full_path + 'data/cropped_angles/' + card)
    height, width, channel = img.shape

    top_image = img[10:height/2, 0:width]
    bottom_image = img[height/2:height-10, 0:width]

    if(cm.have_space(top_image)):
        top_image = img[0:height/2 - 5, 0:width]
    else:
        top_image = img[0:height/2 + 5, 0:width]

    if(cm.have_space(bottom_image)):
        bottom_image = img[height/2 + 5:height, 0:width]
    else:
        bottom_image = img[height/2 - 5:height, 0:width]

    top = Image.fromarray(top_image)
    bottom = Image.fromarray(bottom_image)
    image_top = top.resize((95, 35), Image.ANTIALIAS)
    image_bottom = bottom.resize((95, 35), Image.ANTIALIAS)

    image_top.save(os.path.join(full_path + 'data/processed_cards', "Top " + str(iterator) + card), 'JPEG', quality=90)
    image_bottom.save(os.path.join(full_path + 'data/processed_cards', "Bottom " + str(iterator) + card), 'JPEG', quality=90)
    iterator += 1
