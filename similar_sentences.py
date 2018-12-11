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
import itertools
from itertools import izip

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return izip(a, a)
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
    stories = []
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        stories.append({'story': story, 'highlights': highlights, 'filename': name})

    return stories

def find_most_two_similar_sentences(stories, embed, session, output_dir):
    story_messages=[]
    highlights_messages=[]
    filenames = []
    number_of_sentences=[]
    try:

        for i in xrange(len(stories)): #get the first 100 into lists
            story_for_this = sent_tokenize(stories[i]['story'])
            number_of_sentences.append(len(story_for_this))
            story_messages.append(story_for_this)
            highlights_messages.append(stories[i]['highlights'][0])
            filenames.append(stories[i]['filename'])

        #unlist all list
        story_messages_unlist = list(itertools.chain(*story_messages))

        message_embeddings = embed(story_messages_unlist)
        message_embeddings = session.run(message_embeddings)
        #find the embedding for the highlight
        message_embeddings_highlight = embed(highlights_messages)
        message_embeddings_highlight = session.run(message_embeddings_highlight)
        #find the correlation between highlight and story sentences
        start_index = 0
        end_index = number_of_sentences[0]

        for k in xrange(len(number_of_sentences)):

            embedding_this_document = message_embeddings[start_index:end_index]
            corr=np.inner(embedding_this_document, message_embeddings_highlight[k])
            corr_new = corr.tolist()
            index, value = max(enumerate(corr_new), key=operator.itemgetter(1))
            corr_new2 = [0 if x == value else x for x in corr_new]
            index2, value2 = max(enumerate(corr_new2), key=operator.itemgetter(1))

            new_article = "{} {} \n\n@highlight \n\n{}".format(story_messages_unlist[start_index+index], story_messages_unlist[start_index+index2],highlights_messages[k])

            start_index=start_index+number_of_sentences[k]
            end_index=start_index+number_of_sentences[k+1]

            output_file = "{}/{}".format(output_dir, filenames[k])
            with open(output_file,'w') as fp:
                fp.write(new_article)
                logging.info(output_file)
    except Exception as e:
        logging.info(e)
    return None

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("-i", "--dir")
    # parser.add_argument("-o", "--output_dir")
    #
    # args = parser.parse_args()
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
    #                     datefmt='%m-%d %H:%M:%S')
    #
    # directory = args.dir
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()
    directory='/Users/yihe/Desktop/Stanford/yi_can_summarize/data/cnn/stories'
    args.output_dir='/Users/yihe/Desktop/Stanford/yi_can_summarize/data/cnn/sentence_encoding/'
    stories = load_stories(directory)
    logging.info('Loaded Stories %d' % len(stories))
    with tf.Graph().as_default():
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/1"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
        embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 18
        config.inter_op_parallelism_threads = 18

        for i in xrange(len(stories)):
            try:
                story_100 = stories[i*100:(i+1)*100]
                logging.info("from {} to {}".format(i*100,(i+1)*100))

                with tf.Session(config=config) as session:
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

                    find_most_two_similar_sentences(story_100, embed=embed, session=session,
                                                output_dir=args.output_dir)
            except Exception as e:
                logging.info(e)
                break