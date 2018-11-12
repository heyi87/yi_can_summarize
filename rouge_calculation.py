"""
Calculate Rouge score between reference and decode
Usage:
python rouge_calculation.py --reference /path/to/ref --summary /path/to/summary decoded
"""

from pythonrouge.pythonrouge import Pythonrouge
from nltk.tokenize import sent_tokenize
import re
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('reference', '', 'file path to reference')
tf.app.flags.DEFINE_string('summary', '', 'file path to summary/decoded')


def open_file_tokenize(filename):
    '''
    :param filename:
    :return: tokenized sentence of document
    '''
    with open(filename, 'r') as f:
        document = f.read()
        document_parts = ''.join(document.split('\n'))

    sentences = [re.sub(r'output=.+\-\-\s|output=.+(CNN)','',sen) for sen in sent_tokenize(document_parts)]
    return sentences

def same_len(reference, summary):
    #check the two list have same number of sentences

    shortest = min(len(reference), len(summary))

    return reference[0:shortest], summary[0:shortest]


def main(ref, dec):
    assert FLAGS.reference and FLAGS.summary

    reference_org=open_file_tokenize(FLAGS.reference)
    summary_org = open_file_tokenize(FLAGS.summary)

    reference, summary = same_len(reference_org, summary_org)

    #rouge calculation
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=90,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print(score)

if __name__ == '__main__':
    tf.app.run()
