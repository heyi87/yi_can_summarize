#!/usr/bin/env python

import glob
import os
import re
import collections
import json

import argparse
import logging

def clean_text(document):
    '''
    take the first two lines of the article and concate
    take the first highlight bullet point
    '''

    i=0
    document_two=''
    for line in document.split('\n'):
        if i >1:
            break
        if len(line)>2:
            i+=1
            document_two += line.lower()

    text = re.sub(' +',' ',re.sub('[^A-Za-z0-9\s\n]+', ' ', document_two))
    highlight = re.sub(' +',' ',re.sub('\n','',re.sub('[^A-Za-z0-9\s]+', ' ', document.split('@highlight')[1].lower())))

    return text, highlight


def create_vocab_file(vocab_file, counter):
    with open(vocab_file, 'w') as writer:
        i=0
        for word, count in counter.most_common(200000):
            writer.write(word + ' ' + str(count) + '\n')
            i+=1
        writer.write('<s> 0\n')
        writer.write('</s> 0\n')
        writer.write('<UNK> 0\n')
        writer.write('<PAD> 0\n')

    logging.info("{} number of words {}".format(vocab_file, i))


def files_to_data(input_filenames, text_vocab_file, highlight_vocab_file):
    total_output={}
    counter_text = collections.Counter()
    counter_highlight = collections.Counter()

    os.chdir(input_filenames)
    i=0
    for filename in glob.glob("*.story"):
        with open(filename, 'r') as f:
            document = f.read()

        text, highlight = clean_text(document)

        total_output[i] =[text, re.sub('\n','',highlight)]
        counter_text.update(text.split())
        counter_highlight.update(highlight.split())
        i+=1
    logging.info("number of files in data.json: {}".format(i))
    logging.info("number of vocab in human_vocab: {}".format(len(counter_text.keys())))
    logging.info("number of vocab in machine_vocab: {}".format(len(counter_highlight.keys())))

    create_vocab_file(text_vocab_file, counter_text)
    create_vocab_file(highlight_vocab_file, counter_highlight)


    return total_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--dir")
    parser.add_argument("-o", "--output_dir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')


    input_files =args.dir
    text_vocab_file = os.path.join(args.output_dir, 'human_vocab.txt')
    highlight_vocab_file=os.path.join(args.output_dir, 'machine_vocab.txt')
    output_json = os.path.join(args.output_dir, 'data.json')

    total_output = files_to_data(input_filenames=input_files, text_vocab_file=text_vocab_file, highlight_vocab_file=highlight_vocab_file)
    with open(output_json, 'w') as fp:
        json.dump(total_output, fp)


