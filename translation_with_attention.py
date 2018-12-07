#!/usr/bin/env python

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
import json
import logging
import argparse

def create_data(json_filename):
    sizes_of_text, sizes_of_summary = [], []
    with open(json_filename, 'r') as f:
        data = json.load(f)
    all_data = []
    for key in data.keys():
        if len(data[key]) > 1:
            all_data.append((data[key][0].strip(), data[key][1].strip()))

            size_of_text = len(data[key][0].strip().split(' '))
            size_of_summary = len(data[key][1].strip().split(' '))
            sizes_of_text.append(size_of_text)
            sizes_of_summary.append(size_of_summary)

    logging.info("number of data points {}".format(len(all_data)))

    Tx, Ty = int(np.percentile(sizes_of_text, 80)), int(np.percentile(sizes_of_summary, 80))

    logging.info("Tx: {}".format(Tx))
    logging.info("Ty: {}".format(Ty))

    return all_data, Tx, Ty

def create_vocab_list(vocab_text):
    vocab_list = {}

    with open(vocab_text, 'r') as vocab_f:
        i = 0
        for line in vocab_f:
            vocab_list[line.split(' ')[0]] = i
            i += 1

    return vocab_list

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab.keys())), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab.keys())), Y)))

    return X, np.array(Y), Xoh, Yoh

def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"

    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"

    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """

    # make lower to standardize
    string = string.split(' ')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, vocab['<UNK>']), string))

    if len(string) < length:
        rep += [vocab['<PAD>']] * (length - len(string))

    # print (rep)
    return rep

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(m, Tx, n_a * 2))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model([X, s0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###

    return context
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-j", "--json_file")
    parser.add_argument("-hu", "--human_vocab")
    parser.add_argument("-ma", "--machine_vocab")


    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')


    data, Tx, Ty = create_data(args.json_file)
    human_vocab, machine_vocab = create_vocab_list(args.human_vocab), create_vocab_list(args.machine_vocab)
    inv_machine = dict(enumerate(sorted(machine_vocab)))

    logging.info("number of data: {}".format(len(data)))
    logging.info("number of vocab in human vocab {}".format(len(human_vocab.keys())))
    logging.info("number of vocab in machine vocab {}".format(len(machine_vocab.keys())))

    X, Y, Xoh, Yoh = preprocess_data(data, human_vocab, machine_vocab, Tx, Ty)

    logging.info("X.shape:", X.shape)
    logging.info("Y.shape:", Y.shape)
    logging.info("Xoh.shape:", Xoh.shape)
    logging.info("Yoh.shape:", Yoh.shape)

    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(softmax,
                           name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=1)

    n_a = 32
    n_s = 64
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(len(machine_vocab), activation=softmax)
    Input(shape=(n_s,), name='s0')
    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

    logging.info("{}".format(model.summary()))

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    m=len(data)
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

    model.save('model.h5')








