import numpy as np
import pickle, sys, os
from ExtractData import Corpus
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32


with open('PickledData/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    train_word = train_data.word2int
    train_token = train_data.token2int
    train_sentences = train_data.sentences2int
    train_tokens = train_data.tokens2int
with open('PickledData/validation_data.pkl', 'rb') as f:
    validation_data = pickle.load(f)
    validation_word = validation_data.word2int
    validation_token = validation_data.token2int
    validation_sentences = validation_data.sentences2int
    validation_tokens = validation_data.tokens2int
with open('PickledData/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    test_word = test_data.word2int
    test_token = test_data.token2int
    test_sentences = test_data.sentences2int
    test_tokens = test_data.tokens2int

def generator(sentences, tokens, n_tokens, batch_size=BATCH_SIZE):
    num_samples = len(sentences)

    while True:

        for offset in range(0, num_samples, batch_size):
            X = sentences[offset:offset + batch_size]
            y = tokens[offset:offset + batch_size]

            y = to_categorical(y, num_classes=n_tokens)

            yield shuffle(X, y)

n_tokens = len(train_token)
n_words = len(train_word)

train_words_sample = pad_sequences(train_sentences, maxlen=MAX_SEQUENCE_LENGTH)
train_tokens_sample = pad_sequences(train_tokens, maxlen=MAX_SEQUENCE_LENGTH)


train_words_sample, train_tokens_sample = shuffle(train_words_sample, train_tokens_sample)


print("sentences for train: ", len(train_sentences))
print("tokens for train: ", len(train_tokens))
print("sentences for validation: ", len(validation_sentences))
print("tokens for validation: ", len(validation_tokens))
print("sentences for test: ", len(test_sentences))
print("tokens for test: ", len(test_tokens))


# make generators for training and validation
train_generator = generator(sentences=train_sentences, tokens=train_tokens, n_tokens=n_tokens + 1)
validation_generator = generator(sentences=validation_sentences, tokens=validation_tokens, n_tokens=len(validation_token) + 1)


with open('PickledData/Glove.pkl', 'rb') as f:
	glove_vocabulary = pickle.load(f)

print('Total %s word vectors.' % len(glove_vocabulary))


train_words_set = set(train_word.keys())
validation_words_set = set(validation_word.keys())
test_words_set = set(test_word.keys())
total_words_set = set.union(train_words_set, validation_words_set, test_words_set)


# + 1 to include the unkown word
embedding_matrix = np.random.random((len(total_words_set) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[i] = embedding_vector

