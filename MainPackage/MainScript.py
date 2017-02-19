import h5py
from keras.models import load_model
import pickle as pick
from skimage.io import imread
import CardRecognition as cd
import ClassificationPokerHand as cph
import os
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

modelNeural = load_model('CardNeuralPoker.h5')
modelRandomForest = pick.load(open("RandomForestPokerHand.bin",'rb'))
allPictures = os.listdir('TestCards')
allPictures = natural_sort(allPictures)
f = open('Results.txt','w')
for picture in allPictures:
    img = imread('TestCards/'+picture)
    list = cd.CardRecognition(img, modelNeural)
    hand = cph.getPokerHand(modelRandomForest, list)
    if(hand == 0):
        f.write(picture+' --- Nothing in hand; not a recognized poker hand.\n')
    elif(hand == 1):
        f.write(picture+' --- One pair; one pair of equal ranks within five cards.\n')
    elif(hand == 2):
        f.write(picture+' --- Two pairs; two pairs of equal ranks within five cards.\n')
    elif(hand == 3):
        f.write(picture+' --- Three of a kind; three equal ranks within five cards.\n')
    elif(hand == 4):
        f.write(picture+' --- Straight; five cards, sequentially ranked with no gaps.\n')
    elif(hand == 5):
        f.write(picture+' --- Flush; five cards with the same suit.\n')
    elif(hand == 6):
        f.write(picture+' --- Full house; pair + different rank three of a kind.\n')
    elif(hand == 7):
        f.write(picture+' --- Four of a kind; four equal ranks within five cards.\n')
    elif(hand == 8):
        f.write(picture+' --- Straight flush; straight + flush.\n')
    elif(hand == 9):
        f.write(picture+' --- Royal flush; {Ace, King, Queen, Jack, Ten} + flush.\n')
    else:
        print 'Error: Nothing found!'
f.close()