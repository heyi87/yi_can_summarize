# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize
import operator
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from os import listdir
import argparse
import logging

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename)
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


# load all stories in a directory
def load_stories(directory):
    stories = {}
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)

        stories[name] = {'story': story, 'highlights': highlights}

    return stories

def find_most_two_similar_sentences(stories, story_index, embed, session, output_dir):
    try:
        #split into sentences
        all_sent = sent_tokenize(stories[story_index]['story'])
        #find the embeddings for all sentences
        message_embeddings = embed(all_sent)
        message_embeddings = session.run(message_embeddings)
        #find the embedding for the highlight
        message_embeddings_highlight = embed([stories[story_index]['highlights'][0]])
        message_embeddings_highlight = session.run(message_embeddings_highlight)
        #find the correlation between highlight and story sentences
        corr = np.inner(message_embeddings_highlight, message_embeddings)
        #find the most similar two sentences to highlight
        corr_new = corr.tolist()[0]
        #print corr_new
        index, value = max(enumerate(corr_new), key=operator.itemgetter(1))
        corr_new2 = [0 if x==value else x for x in corr_new]
        index2, value2 = max(enumerate(corr_new2), key=operator.itemgetter(1))
        new_article ="{} {} \n\n@highlight \n\n{}".format(all_sent[index], all_sent[index2], stories[story_index]['highlights'][0])

        output_file = "{}/{}".format(output_dir, story_index)
        with open(output_file,'w') as fp:
            fp.write(new_article)
            logging.info(output_file)
    except Exception as e:
        logging.info(e)
    return story_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--dir")
    parser.add_argument("-o", "--output_dir")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    directory = args.dir
    stories = load_stories(directory)
    logging.info('Loaded Stories %d' % len(stories))

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/1"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
    embed = hub.Module(module_url)
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 18
    config.inter_op_parallelism_threads = 18

    with tf.Session(config=config) as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        for i in stories:
            find_most_two_similar_sentences(stories, story_index=i, embed=embed, session=session, output_dir=args.output_dir)
            logging.info(i)