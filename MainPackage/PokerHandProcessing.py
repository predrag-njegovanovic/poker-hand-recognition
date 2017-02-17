import os

f = open('Soft-dataset-pokerHand/PokerHand-Original')
textLines = []
for line in f:
    newData = []
    parts = line.split(",")
    for i in xrange(0,len(parts),2):
        if i == 10:
            newData.append(parts[i])
            break
        rank = 0
#        print parts[i]
        if(int(parts[i]) == 1):
            rank = 15
        elif(int(parts[i]) == 2):
            rank = 17
        elif(int(parts[i]) == 3):
            rank = 16
        elif(int(parts[i]) == 4):
            rank = 18
        newData.append(rank)
        newData.append(parts[i+1])
    textLines.append(newData)

w = open('Soft-dataset-pokerHand/PokerHandDataset.txt','w')
for line in textLines:
    lineToWrite = ",".join(map(str,line))
    w.write(lineToWrite)

