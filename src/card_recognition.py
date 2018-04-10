import cv2
import copy
import operator
import numpy as np
import neural_network as nnwd
import card_manipulation as cm
import matplotlib.pyplot as plt


def card_recognition(img, model):
    c = copy.copy(img)
    counter = 7
    first_counter = 0
    image_array = []
    contour_array = []
    while first_counter < 100:
        contours_of_four = []
        counter += 1
        i = cm.process_image(img, counter)
        image, contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(c, contours, -1, (255, 255, 0), 2)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        for cons in contour:
            p = cv2.arcLength(cons, True)
            a = cv2.approxPolyDP(cons, 0.02 * p, True)
            if(len(a) == 4):
                contours_of_four.append(cons)
            if(len(contours_of_four) > 8 and len(a) > 4):
                break

        x = 0
        contour_list = []
        contour_list.append(contours_of_four[0])
        while x < len(contours_of_four):
            f = cv2.moments(contours_of_four[x])
            cX = int(f["m10"]/f["m00"])
            cY = int(f["m01"]/f["m00"])
            flag = False
            j = 0
            while j < len(contour_list):
                v = cv2.moments(contour_list[j])
                vX = int(v["m10"] / v["m00"])
                vY = int(v["m01"] / v["m00"])
                j += 1
                if(abs(vX - cX) >= 0 and abs(vX - cX) < 5 and abs(vY - cY) >= 0 and abs(vY - cY) < 5):
                    flag = True
                    break
            if not flag:
                contour_list.append(contours_of_four[x])
                if(len(contour_list) == 6):
                    break
            x += 1

        flag = False
        contour_array = []

        for idx, c in enumerate(contour_list):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if(len(approx) > 4):
                flag = True
            list = []
            for idx, i in enumerate(approx):
                list.append(approx[idx][0])

            contour_array.append(list)
        first_counter += 1
        if not flag:
            break

    for index, list in enumerate(contour_array):
        if(index > 4):
            break
        approx = np.array(list, np.float32)
        dst = np.array([[0, 0], [0, 499], [499, 499], [499, 0]], np.float32)
        M = cv2.getPerspectiveTransform(approx, dst)
        z = cv2.warpPerspective(img, M, (500, 500))
        image_array.append(z)

    crop_cards = []
    for image in image_array:
        for num_angle in range(4):
            if(num_angle == 0):
                crops = image[0:120, 0:120]
            elif(num_angle == 1):
                crops = image[380:500, 0:120]
            elif(num_angle == 2):
                crops = image[0:120, 380:500]
            else:
                crops = image[380:500, 380:500]

            sec_counter = 2
            first_counter = 0
            angle_rotate = -90
            while first_counter < 20:
                whites = 0
                blacks = 0
                tO = cm.suit(crops, sec_counter, angle_rotate)
                h, w, c = tO.shape
                gray = cv2.cvtColor(tO, cv2.COLOR_BGR2GRAY)
                filter = np.ones((5, 5), np.float32) / 25
                i = cv2.filter2D(gray, -1, filter)
                i = cv2.adaptiveThreshold(i, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
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

            ht, wt, ct = tO.shape
            if (ht > 115 and wt > 115):
                tO = cm.cut_half(tO)

            crop_cards.append(tO)

    resizes = []
    for card in crop_cards:
        s = cv2.resize(card, (95, 70), interpolation=cv2.INTER_CUBIC)
        resizes.append(s)

    picture_parts = []

    for picture in resizes:

        height, width, channel = picture.shape

        top_image = picture[10:height // 2, 0:width]
        bottom_image = picture[height // 2:height - 10, 0:width]

        if (cm.have_space(top_image)):
            top_image = picture[0:height // 2 - 5, 0:width]
        else:
            top_image = picture[0:height // 2 + 5, 0:width]

        if (cm.have_space(bottom_image)):
            bottom_image = picture[height // 2 + 5:height, 0:width]
        else:
            bottom_image = picture[height // 2 - 5:height, 0:width]

        image_top = cv2.resize(top_image, (95, 35), interpolation=cv2.INTER_CUBIC)
        image_bottom = cv2.resize(bottom_image, (95, 35), interpolation=cv2.INTER_CUBIC)

        picture_parts.append(image_top)
        picture_parts.append(image_bottom)

    accumulation_list = []
    number_list = []

    for r in picture_parts:
        number, probability = nnwd.check_card(model, r)
        t = (number, probability)
        accumulation_list.append(t)
        if(len(accumulation_list) == 8):
            accumulation_list = sorted(accumulation_list,
                                       key=operator.itemgetter(1),
                                       reverse=True)

            number_list.append(accumulation_list[0][0])
            flag = accumulation_list[0][0]
            for x, _ in enumerate(accumulation_list):
                if(flag >= 15 and accumulation_list[x][0] < 15):
                    number_list.append(accumulation_list[x][0])
                    break
                elif(flag < 15 and accumulation_list[x][0] >= 15):
                    number_list.append(accumulation_list[x][0])
                    break

            accumulation_list = []

    final_list = []
    for index in range(0, len(number_list)-1, 2):
        suit = number_list[index]
        number = number_list[index+1]
        if suit >= 15:
            final_list.append(suit)
            final_list.append(number)
        else:
            final_list.append(suit)
            final_list.append(number)

    return final_list
