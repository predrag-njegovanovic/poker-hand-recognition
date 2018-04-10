import os
import card_manipulation as cm

from PIL import Image
from skimage.io import imread


allPictures = os.listdir('Soft-dataset-uglovi')
iterator = 0
for card in allPictures:
    img = imread('Soft-dataset-uglovi/' + card)
    height, width, channel = img.shape

    TopImage = img[10:height/2, 0:width]
    BottomImage = img[height/2:height-10, 0:width]

    if(cm.haveSpace(TopImage)):
        TopImage = img[0:height/2 - 5, 0:width]
    else:
        TopImage = img[0:height/2 + 5, 0:width]

    if(cm.haveSpace(BottomImage)):
        BottomImage = img[height/2 + 5:height, 0:width]
    else:
        BottomImage = img[height/2 - 5:height, 0:width]

    top = Image.fromarray(TopImage)
    bottom = Image.fromarray(BottomImage)
    imageTop = top.resize((95, 35), Image.ANTIALIAS)
    imageBottom = bottom.resize((95, 35), Image.ANTIALIAS)
    imageTop.save(os.path.join('Soft-dataset-duplicated', "Top " + str(iterator) + card), 'JPEG', quality=90)
    imageBottom.save(os.path.join('Soft-dataset-duplicated', "Bottom " + str(iterator) + card), 'JPEG', quality=90)
    iterator += 1
