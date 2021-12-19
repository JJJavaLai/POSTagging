# import necessary libraries


import numpy as np
import pandas as pd
from RnnModel import POSTaggingModel

from keras.utils.np_utils import to_categorical

import pickle

with open('PickledData/V2.pkl', 'rb') as f:
    V2 = pickle.load(f)
    print("V2 loaded, embedded shape: ", V2.shape)

with open('PickledData/V3.pkl', 'rb') as f:
    V3 = pickle.load(f)
    print("V3 loaded, embedded shape: ", V3.shape)

with open('PickledData/V4.pkl', 'rb') as f:
    V4 = pickle.load(f)
    print("V4 loaded, embedded shape: ", V4.shape)

# "train_X": train_X,
# "train_Y": train_Y,
# "validation_X": validation_X,
# "validation_Y": validation_Y,
# "test_X": test_X,
# "test_Y": test_Y,
# "MAX_SEQ_LENGTH": 100

with open('PaddedData/padded_samples.pkl', 'rb') as f:
    samples = pickle.load(f)
    print("samples loaded, embedded shape: ")
    train_X = samples["train_X"]
    print("sample for training(train_X): ", len(train_X))
    train_Y = samples["train_Y"]
    print("sample for training(train_Y): ", len(train_Y))
    validation_X = samples["validation_X"]
    print("sample for validation(validation_X): ", len(validation_X))
    validation_Y = samples["validation_Y"]
    print("sample for validation(validation_Y): ", len(validation_Y))
    test_X = samples["test_X"]
    print("sample for test(test_X): ", len(test_X))
    test_Y = samples["test_Y"]
    print("sample for test(test_Y): ", len(test_Y))
    MAX_SEQ_LENGTH = samples["MAX_SEQ_LENGTH"]
    print("MAX_SEQ_LENGTH: ", MAX_SEQ_LENGTH)
    int2token = samples["int2token"]
    int2word = samples["int2word"]

with open('PickledData/pickled_data.pkl', 'rb') as f:
    pickled_data = pickle.load(f)
    train_VOCABULARY_SIZE = len(pickled_data["train_words"])
NUM_CLASSES = len(int2token) + 1
VOCABULARY_SIZE = train_VOCABULARY_SIZE
EMBEDDING_SIZE = 300
embedding_weights = V2
batch_size = 32
epoch = 20

print("TRAINING DATA")
print('Shape of input sequences: {}'.format(train_X.shape))
print('Shape of output sequences: {}'.format(train_Y.shape))
print("-" * 50)
print("VALIDATION DATA")
print('Shape of input sequences: {}'.format(validation_X.shape))
print('Shape of output sequences: {}'.format(validation_Y.shape))
print("-" * 50)
print("TESTING DATA")
print('Shape of input sequences: {}'.format(test_X.shape))
print('Shape of output sequences: {}'.format(test_Y.shape))

optimizer_list = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
loss_functions = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                 "mean_squared_logarithmic_error", "squared_hinge",
                 "hinge", "categorical_hinge", "logcosh"]
train_Y = to_categorical(train_Y)
validation_Y = to_categorical(validation_Y)
# Train Bidirectional_LSTM_Model
time = 0
for optimizer in optimizer_list:
    for loss in loss_functions:
        time += 1
        name = "Bidirectional_LSTM_Model_" + str(time)
        Bidirectional_LSTM_Model = POSTaggingModel(NUM_CLASSES, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH,
                                                   train_X,
                                                   train_Y, validation_X, validation_Y, embedding_weights, batch_size,
                                                   epoch, name, loss, optimizer)
        Bidirectional_LSTM_Model.buildModel("Bidirectional LSTM Model")
        Bidirectional_LSTM_Model.fitModel()
        Bidirectional_LSTM_Model.evaluateModel(test_X, test_Y)

        # Train LSTM_2_Dense_Model
        name = "LSTM_2_Dense_Model_" + str(time)
        LSTM_2_Dense_Model = POSTaggingModel(NUM_CLASSES, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, train_X,
                                             train_Y, validation_X, validation_Y, embedding_weights, batch_size, epoch, name, loss, optimizer)
        LSTM_2_Dense_Model.buildModel("Bidirectional LSTM Model with 2 Dense")
        LSTM_2_Dense_Model.fitModel()
        LSTM_2_Dense_Model.evaluateModel(test_X, test_Y)

        # Train B2_LSTM_Model
        name = "B2_LSTM_Model_" + str(time)
        B2_LSTM_Model = POSTaggingModel(NUM_CLASSES, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, train_X,
                                        train_Y, validation_X, validation_Y, embedding_weights, batch_size, epoch, name, loss, optimizer)
        B2_LSTM_Model.buildModel("2 LSTM Model")
        B2_LSTM_Model.fitModel()
        B2_LSTM_Model.evaluateModel(test_X, test_Y)

        # Train Gru_Model
        name = "Gru_Model_" + str(time)
        Gru_Model = POSTaggingModel(NUM_CLASSES, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, train_X,
                                    train_Y, validation_X, validation_Y, embedding_weights, batch_size, epoch, name, loss, optimizer)
        Gru_Model.buildModel("GRU Model")
        Gru_Model.fitModel()
        Gru_Model.evaluateModel(test_X, test_Y)





# # Train LSTM_Model
# name = "LSTM_Model" + time
# LSTM_Model = POSTaggingModel(NUM_CLASSES, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, train_X,
#                              train_Y, validation_X, validation_Y, embedding_weights, batch_size, epoch)
# LSTM_Model.buildModel("LSTM Model")
# LSTM_Model.fitModel()
# LSTM_Model.evaluateModel(test_X, test_Y)


