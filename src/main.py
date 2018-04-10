import os
import re
import pickle as pick
import card_recognition as cd
import poker_hand_classifier as cph

from skimage.io import imread
from keras.models import load_model


full_path = os.path.dirname(__file__)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == "__main__":
    f = open(full_path + '/results/results.txt', 'w')
    c = open(full_path + '/results/cards.txt', 'w')
    neural_net = load_model(full_path + '/models/neural_model.h5')
    random_forest = pick.load(open(full_path + "/models/random_forest.pkl", 'rb'))

    pictures = os.listdir(full_path + '/data/test_dataset/')
    sorted_pictures = natural_sort(pictures)
    print('--- Running through test images...')
    for picture in sorted_pictures:
        print(picture)
        img = imread(full_path + '/data/test_dataset/'+picture)
        card_list = cd.card_recognition(img, neural_net)
        c.write(str(card_list)+'\n')
        hand = cph.get_poker_hand(random_forest, card_list)
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
            print('Error: Nothing found!')
    f.close()
    c.close()
    print('--- Results have been written to results folder')
