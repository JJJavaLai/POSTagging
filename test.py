
from RnnModel import POSTaggingModel
import numpy as np
import pickle, sys, os
from ExtractData import Corpus
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, recall_score

MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 300
BATCH_SIZE = 32

with open('PickledData/total_data.pkl', 'rb') as f:
    total_data = pickle.load(f)
    total_word = total_data.word2int
    total_token = total_data.token2int
    total_sentences = total_data.sentences2int
    total_tokens = total_data.tokens2int
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





N_TOKENS = len(total_token)
n_words = len(total_word)

print("Total words: ", n_words)
print("Total tokens: ", N_TOKENS)
print("Total sentences: ", )



train_sentences_sample = pad_sequences(train_sentences, maxlen=MAX_SEQUENCE_LENGTH)
train_tokens_sample = pad_sequences(train_tokens, maxlen=MAX_SEQUENCE_LENGTH)
validation_sentences_sample = pad_sequences(validation_sentences, maxlen=MAX_SEQUENCE_LENGTH)
validation_tokens_sample = pad_sequences(validation_tokens, maxlen=MAX_SEQUENCE_LENGTH)
test_sentences_sample = pad_sequences(test_sentences, maxlen=MAX_SEQUENCE_LENGTH)
test_tokens_sample = pad_sequences(test_tokens, maxlen=MAX_SEQUENCE_LENGTH)

train_sentences_sample, train_tokens_sample = shuffle(train_sentences_sample, train_tokens_sample)

print("sentences for train: ", len(train_sentences))
print("tokens for train: ", len(train_tokens))
print("sentences for validation: ", len(validation_sentences))
print("tokens for validation: ", len(validation_tokens))
print("sentences for test: ", len(test_sentences))
print("tokens for test: ", len(test_tokens))

# make generators for training and validation
# train_generator = generator(sentences=train_sentences_sample, tokens=train_tokens_sample, n_tokens=n_tokens + 1)
# validation_generator = generator(sentences=validation_sentences_sample, tokens=validation_tokens_sample,
#                                  n_tokens=len(validation_token) + 1)

with open('PickledData/Glove.pkl', 'rb') as f:
    glove_vocabulary = pickle.load(f)

print('Total %s word vectors.' % len(glove_vocabulary))

# + 1 to include the unkown word
embedding_matrix = np.random.random((n_words + 1, EMBEDDING_DIM))

for word, i in total_word.items():
    embedding_vector = glove_vocabulary.get(word)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', train_sentences_sample.shape)


VOCABULARY_SIZE = n_words + 1
EMBEDDING_SIZE = EMBEDDING_DIM
MAX_SEQ_LENGTH = MAX_SEQUENCE_LENGTH = 150
train_X = train_sentences_sample
train_Y = train_tokens_sample
validation_X = validation_sentences_sample
validation_Y = validation_tokens_sample
embedding_weights = embedding_matrix
batch_size = 32
epoch = 20

model = POSTaggingModel(N_TOKENS, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, train_X,
                        train_Y, validation_X, validation_Y, embedding_weights, batch_size, epoch)
model.buildModel("Bidirectional LSTM Model")
model.fitModel()
model.evaluateModel(test_sentences_sample, test_tokens_sample)