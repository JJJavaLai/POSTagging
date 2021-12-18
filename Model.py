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





n_tokens = len(total_token)
n_words = len(total_word)

print("Total words: ", n_words)
print("Total tokens: ", n_tokens)
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

embedding_layer = Embedding(n_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
preds = TimeDistributed(Dense(n_tokens + 1, activation='softmax'))(l_lstm)
model = Model(sequence_input, preds)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['sparse_categorical_accuracy'])
model.summary()
print("model fitting - Bidirectional LSTM")

model.fit(train_sentences_sample,
          train_tokens_sample,
          batch_size=BATCH_SIZE,
          epochs=500,
          validation_data=(validation_sentences_sample, validation_tokens_sample))

if not os.path.exists('Models/'):
    print('MAKING DIRECTORY Models/ to save model file')
    os.makedirs('Models/')

train = True

if train:
    model.save('Models/model.h5')
    print('MODEL SAVED in Models/ as model.h5')
else:
    from keras.models import load_model

    model = load_model('Models/model_in_sentence.h5')


# predict_result = np.argmax(model.predict(test_sentences_sample), axis = 2)
# model_accuracy_score = accuracy_score(test_tokens_sample, predict_result)
# print("The accuracy of model: ", model_accuracy_score)
test_results = model.evaluate(test_sentences_sample, test_tokens_sample, verbose=0)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))
# model_f1_score = f1_score(test_tokens_sample, predict_result, labels=None, average='binary', sample_weight=None)
# print("The F1 score of model: ", model_f1_score)
# model_recall_score = recall_score(test_tokens_sample, predict_result, labels=None, pos_label=1, average='binary', sample_weight=None)
# print("The recall score of model: ", model_recall_score)
# print("Classification report of model:")
# print(classification_report(test_tokens_sample, predict_result, target_names=None, sample_weight=None))




