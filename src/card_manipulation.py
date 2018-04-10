import cv2
import numpy as np
from operator import itemgetter


def have_space(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i,
                              1,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV,
                              11,
                              3)
    i = turn_white(img, i)
    for x in range(len(i)):
        clear_column = True
        for k in range(len(i[x])):
            if (i[x][k] == 1):
                clear_column = False
                break

        if(clear_column):
            return True

    return False


def get_blacks(i):
    whites = 0
    for x in range(len(i)):
        for k in range(len(i[x])):
            if (i[x][k] == 1):
                whites += 1
    return whites


def cut_half(img):
    height, width, channel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i,
                              1,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV,
                              11,
                              3)
    black_image = turn_white(img, i)
    first_column = black_image[:, 5:10]
    last_column = black_image[:, width-10:width-5]
    first_row = black_image[5:10, :]
    last_row = black_image[height-10:height-5, :]

    top_image = img[0:int(height/2)-20, 0:width]
    bottom_image = img[int(height/2)+20:height, 0:width]
    left_image = img[0:height, 0:int(width/2)+20]
    right_image = img[0:height, int(width/2)-20:width]

    fC = get_blacks(first_column)
    lC = get_blacks(last_column)
    fR = get_blacks(first_row)
    lR = get_blacks(last_row)
    temp = ((fC, right_image),
            (lC, left_image),
            (fR, bottom_image),
            (lR, top_image))
    pom = sorted(temp, key=itemgetter(0))[-1]
    im = pom[1]
    res = im
    height, width, channel = im.shape
    if(width > height):
        res = cv2.resize(im, (120, 120), interpolation=cv2.INTER_CUBIC)
        mat = cv2.getRotationMatrix2D((60, 60), -90, 1)
        im = cv2.warpAffine(res, mat, (120, 120))
        res = cv2.resize(im, (height, width), interpolation=cv2.INTER_CUBIC)
    return res


def find_contours(img):
    image, contours, hierarchy = cv2.findContours(img,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours,
                     key=cv2.contourArea,
                     reverse=True)
    list = []

    t = cv2.boundingRect(contour[0])
    list.append(t)
    max = 0
    idx = 0
    for index, i in enumerate(list):
        if (i[3] > max):
            max = i[3]
            idx = index

    x = list[idx][0]
    y = list[idx][1]
    w = list[idx][2]
    h = list[idx][3]
    return x, y, w, h


def suit(crops_top, counter, angle_rotate):
    crops = process_image(crops_top, counter)
    x, y, w, h = find_contours(crops)
    top_out = crops_top[y:y+h, x:x+w]
    crop_height, crop_width, channel = crops_top.shape
    height, width, channel = top_out.shape
    if(width > height):
        mat = cv2.getRotationMatrix2D((crop_width/2, crop_height/2), angle_rotate, 1)
        new_picture = cv2.warpAffine(crops_top, mat, (crop_width, crop_height))
        cc = process_image(new_picture, counter)
        x, y, w, h = find_contours(cc)
        cv2.rectangle(cc, (x, y), (x + w, y + h), (255, 255, 0), 2)
        top_out = new_picture[y:y+h, x:x+w]

    return top_out


def process_image(img, nmbOfDilate):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter = np.ones((5, 5), np.float32) / 25
    i = cv2.filter2D(gray, -1, filter)
    i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    kernel = np.ones((3, 3), np.uint8)

    i = cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel, iterations=1)
    i = cv2.dilate(i, kernel, iterations=nmbOfDilate)
    return i


def turn_white(img, black_img):
    h, w, c = img.shape
    for x in range(h):
        for j in range(w):
            if((img[x][j][0] > 120 and img[x][j][1] < 80 and img[x][j][2] < 80) or (img[x][j][0] < 60 and img[x][j][1] < 60 and img[x][j][2] < 60)):
                black_img[x][j] = 1

    return black_img
