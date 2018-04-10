import os
import cv2
import numpy as np
import card_manipulation as cm

from PIL import Image
from skimage.io import imread

full_path = os.path.dirname(__file__)

pictures = os.listdir(full_path + '/data/playing_cards_dataset/')
for card in pictures:
    img = imread(full_path + '/data/playing_cards_dataset/'+card)
    counter = 7
    first_counter = 0
    while first_counter < 100:
        counter += 2
        processed_image = cm.process_image(img, counter)
        image, contours, _ = cv2.findContours(processed_image,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        c = contour[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        approx_list = []
        for idx, i in enumerate(approx):
            approx_list.append(approx[idx][0])

        first_counter += 1
        if(len(approx) <= 4):
            break

    approx = np.array(approx_list, np.float32)
    dst = np.array([[0, 0], [0, 499], [499, 499], [499, 0]], np.float32)
    perspective = cv2.getPerspectiveTransform(approx, dst)
    warp_image = cv2.warpPerspective(img, perspective, (500, 500))
    crops_top = warp_image[0:120, 0:120]
    crops_bottom = warp_image[380:500, 0:120]
    angle_rotate = -90
    sec_counter = 2
    first_counter = 0
    while first_counter < 20:
        whites = 0
        blacks = 0
        tO = cm.suit(crops_top, sec_counter, angle_rotate)
        h, w, c = tO.shape
        gray = cv2.cvtColor(tO, cv2.COLOR_BGR2GRAY)
        filter = np.ones((5, 5), np.float32) / 25
        i = cv2.filter2D(gray, -1, filter)
        i = cv2.adaptiveThreshold(i,
                                  1,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,
                                  11,
                                  3)
        for x in range(len(i)):
            for k in range(len(i[x])):
                if(i[x][k] == 0):
                    blacks += 1
                else:
                    whites += 1

        first_counter += 1
        sec_counter += 1
        if(blacks/whites > 5):
            continue

        if(h > 80 and h < 110 and w > 40 and w < 100):
            break

    # Crops bottom
    sec_counter = 2
    first_counter = 0
    angle_rotate = 90
    while first_counter < 20:
        whites = 0
        blacks = 0
        bO = cm.suit(crops_bottom, sec_counter, angle_rotate)
        h, w, c = bO.shape
        gray = cv2.cvtColor(bO, cv2.COLOR_BGR2GRAY)
        filter = np.ones((5, 5), np.float32) / 25
        i = cv2.filter2D(gray, -1, filter)
        i = cv2.adaptiveThreshold(i,
                                  1,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,
                                  11,
                                  3)
        for x in range(len(i)):
            for k in range(len(i[x])):
                if (i[x][k] == 0):
                    blacks += 1
                else:
                    whites += 1

        first_counter += 1
        sec_counter += 1
        if (blacks / whites > 5):
            continue

        if (h > 80 and h < 110 and w > 40 and w < 100):
            break

    ht, wt, ct = tO.shape
    if(ht > 115 and wt > 115):
        tO = cm.cut_half(tO)

    hb, wb, cb = bO.shape
    if(hb > 115 and wb > 115):
        bO = cm.cut_half(bO)

    top = Image.fromarray(tO)
    bottom = Image.fromarray(bO)
    image_top = top.resize((95, 70), Image.ANTIALIAS)
    image_bottom = bottom.resize((95, 70), Image.ANTIALIAS)

    image_top.save(os.path.join(full_path + '/data/cropped_angles/', "Top "+card),
                   'JPEG',
                   quality=90)
    image_bottom.save(os.path.join(full_path + '/data/cropped_angles/', "Bottom "+card),
                      'JPEG',
                      quality=90)
