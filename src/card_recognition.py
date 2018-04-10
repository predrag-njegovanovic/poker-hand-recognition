import cv2
import copy
import operator
import numpy as np
import neural_network as nnwd
import card_manipulation as cm


def CardRecognition(img, model):
    c = copy.copy(img)
    counter = 7
    wFirstCounter = 0
    arrayOfImages = []
    arrayOfContures = []
    while wFirstCounter < 100:
        contoursOf4 = []
        counter += 1
        i = cm.obradiSliku(img, counter)
        image, konture, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(c, konture, -1, (255, 255, 0), 2)
        kontura = sorted(konture, key=cv2.contourArea, reverse=True)[:10]
        for cons in kontura:
            p = cv2.arcLength(cons, True)
            a = cv2.approxPolyDP(cons, 0.02 * p, True)
            if(len(a) == 4):
                contoursOf4.append(cons)
            if(len(contoursOf4) > 8 and len(a) > 4):
                break

        x = 0
        listOfContours = []
        listOfContours.append(contoursOf4[0])
        while x < len(contoursOf4):
            f = cv2.moments(contoursOf4[x])
            cX = int(f["m10"]/f["m00"])
            cY = int(f["m01"]/f["m00"])
            flag = False
            j = 0
            while j < len(listOfContours):
                v = cv2.moments(listOfContours[j])
                vX = int(v["m10"] / v["m00"])
                vY = int(v["m01"] / v["m00"])
                j += 1
                if(abs(vX - cX) >= 0 and abs(vX - cX) < 5 and abs(vY - cY) >= 0 and abs(vY - cY) < 5):
                    flag = True
                    break
            if not flag:
                listOfContours.append(contoursOf4[x])
                if(len(listOfContours) == 6):
                    break
            x += 1

        flag = False
        arrayOfContures = []

        for idx, c in enumerate(listOfContours):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if(len(approx) > 4):
                flag = True
            list = []
            for idx, i in enumerate(approx):
                list.append(approx[idx][0])

            arrayOfContures.append(list)
        wFirstCounter += 1
        if not flag:
            break

    for index, list in enumerate(arrayOfContures):
        if(index > 4):
            break
        approx = np.array(list, np.float32)
        dst = np.array([[0, 0], [0, 499], [499, 499], [499, 0]], np.float32)
        M = cv2.getPerspectiveTransform(approx, dst)
        z = cv2.warpPerspective(img, M, (500, 500))
        arrayOfImages.append(z)

    cropCards = []
    for image in arrayOfImages:
        for numAngle in range(4):
            if(numAngle == 0):
                crops = image[0:120, 0:120]
            elif(numAngle == 1):
                crops = image[380:500, 0:120]
            elif(numAngle == 2):
                crops = image[0:120, 380:500]
            else:
                crops = image[380:500, 380:500]

            secCounter = 2
            wFirstCounter = 0
            angleRotate = -90
            while wFirstCounter < 20:
                whites = 0
                blacks = 0
                tO = cm.Znak(crops, secCounter, angleRotate)
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

                wFirstCounter += 1
                secCounter += 1
                if(blacks/whites > 5):
                    continue
                if(h > 80 and h < 110 and w > 40 and w < 100):
                    break

            ht, wt, ct = tO.shape
            if (ht > 115 and wt > 115):
                tO = cm.cutInHalf(tO)

            cropCards.append(tO)

    resizesFirst = []
    for card in cropCards:
        s = cv2.resize(card, (95, 70), interpolation=cv2.INTER_CUBIC)
        resizesFirst.append(s)

    resizes = []
    for slika in resizesFirst:

        height, width, channel = slika.shape

        TopImage = slika[10:height / 2, 0:width]
        BottomImage = slika[height / 2:height - 10, 0:width]

        if (cm.haveSpace(TopImage)):
            TopImage = slika[0:height / 2 - 5, 0:width]
        else:
            TopImage = slika[0:height / 2 + 5, 0:width]

        if (cm.haveSpace(BottomImage)):
            BottomImage = slika[height / 2 + 5:height, 0:width]
        else:
            BottomImage = slika[height / 2 - 5:height, 0:width]

        imageTop = cv2.resize(TopImage, (95, 35), interpolation=cv2.INTER_CUBIC)
        imageBottom = cv2.resize(BottomImage, (95, 35), interpolation=cv2.INTER_CUBIC)

        resizes.append(imageTop)
        resizes.append(imageBottom)

    accumulationList = []
    listOfNumbers = []

    for r in resizes:
        number, probability = nnwd.checkCard(model, r)
        t = (number, probability)
        accumulationList.append(t)
        if(len(accumulationList) == 8):
            accumulationList = sorted(accumulationList, key=operator.itemgetter(1), reverse=True)

            listOfNumbers.append(accumulationList[0][0])
            flag = accumulationList[0][0]
            for x, _ in enumerate(accumulationList):
                if(flag >= 15 and accumulationList[x][0] < 15):
                    listOfNumbers.append(accumulationList[x][0])
                    break
                elif(flag < 15 and accumulationList[x][0] >= 15):
                    listOfNumbers.append(accumulationList[x][0])
                    break

            accumulationList = []

    finalList = []
    for index in range(0, len(listOfNumbers)-1, 2):
        znak = listOfNumbers[index]
        broj = listOfNumbers[index+1]
        if znak >= 15:
            finalList.append(znak)
            finalList.append(broj)
        else:
            finalList.append(broj)
            finalList.append(znak)

    return finalList
