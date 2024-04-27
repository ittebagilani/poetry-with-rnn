import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# read text from file, decode the binary using the utf-8 encoding.
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))
# print(characters)

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i, c in enumerate(characters))

# print(index_to_char)

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []

next_char = []

'''
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])          # this line appends a bunch of characters 
    next_char.append(text[i+SEQ_LENGTH])          # this line adds the next character to the list (the one after where the previous line ends off)


x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    
    y[i, char_to_index[next_char[i]]] = 1
'''
model = tf.keras.models.load_model('poetry.h5')