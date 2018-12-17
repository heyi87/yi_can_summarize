import h5py
import numpy as np
import re
import os
import glob
import argparse
import logging

def get_embedding(numberBatch_emb,word):
    item='/c/en/{}'.format(word)
    index = np.where(numberBatch_emb[u'mat'][u'axis1'].value == item)[0][0]
    return numberBatch_emb[u'mat'][u'block0_values'][index]


def string_to_int(string, length, Numberbatch_emb):

    # make lower to standardize
    string = string.split(' ')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: get_embedding(Numberbatch_emb, x), string))

    if len(string) < length:
        rep += get_embedding(Numberbatch_emb,'pad') * (length - len(string))

    rep = [[i] for i in rep]

    # print (rep)
    return rep

def clean_text(document):

    i=0
    document_two=''
    for line in document.split('\n'):
        if len(line)>2:
            i+=1
            document_two += line.lower()

    text = re.sub(' +',' ',re.sub('[^A-Za-z0-9\s\n]+', ' ', document_two)) #remove special characters and keep only letters
    highlight = re.sub(' +',' ',re.sub('\n','',re.sub('[^A-Za-z0-9\s]+', ' ', document.split('@highlight')[1].lower()))) #remove special characters and keep only letters and first bullet point

    return text, highlight


def files_to_data(input_dir, output_dir, Numberbatch_emb):

    raw_X, raw_Y, num_of_words_in_X = [], [], []

    os.chdir(input_dir)
    for filename in glob.glob("*.story"): #get all the stories from cnn and daily news
        with open(filename, 'r') as f:
            document = f.read()

        text, highlight = clean_text(document)
        raw_X.append(text)
        raw_Y.append(highlight)
        num_of_words_in_X.append(len(text))


    #some stories have too many words, remove the longest 5% of words, take the 95 percentile
    X = np.array([string_to_int(i, max(num_of_words_in_X), Numberbatch_emb) for i in raw_X]) #to np array
    Y = np.array([string_to_int(i, 30, Numberbatch_emb) for i in raw_Y]) #to np array

    X.tofile(os.path.join(output_dir, 'X.dat'))
    Y.tofile(os.path.join(output_dir, 'Y.dat'))

    return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--dir")
    parser.add_argument("-o", "--output_dir")
    parser.add_argument("ne", "--numberBatch_emb", default='mini.h5')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    args = parser.parse_args()

    logging.info("load numberBatch embedding {}".format(args.numberBatch_emb))
    numberBatch_emb = h5py.File(args.numberBatch_emb, "r+")

    files_to_data(input_dir=args.dir, output_dir=args.output_dir, Numberbatch_emb=numberBatch_emb)








