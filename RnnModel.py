from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, recall_score
import numpy as np
class POSTaggingModel:
    def __init__(self, N_TOKENS, VOCABULARY_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, train_X, train_Y,
                 validation_X, validation_Y, embedding_weights, batch_size, epoch):
        self.N_TOKENS = N_TOKENS
        self.VOCABULARY_SIZE = VOCABULARY_SIZE
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.train_X = train_X
        self.train_Y = train_Y
        self.validation_X = validation_X
        self.validation_Y = validation_Y
        self.embedding_weights = embedding_weights
        self.batch_size = batch_size
        self.epoch = epoch
        self.__model_type = ["GRU Model", "Bidirectional LSTM Model", "LSTM Model"]
        self.model = Sequential()

    # baseline
    def __construct_bidirectional_LSTM_model(self):
        self.model.add(Embedding(input_dim=self.VOCABULARY_SIZE,
                                     output_dim=self.EMBEDDING_SIZE,
                                     input_length=self.MAX_SEQ_LENGTH,
                                     weights=[self.embedding_weights],
                                     trainable=False
                                     ))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(self.N_TOKENS, activation='softmax')))

        self.model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['acc'])
        print("Model built")
        self.model.summary()

    def __construct_GRU_model(self):
        gru_model = Sequential()
        gru_model.add(Embedding(input_dim=self.VOCABULARY_SIZE,
                                output_dim=self.EMBEDDING_SIZE,
                                input_length=self.MAX_SEQ_LENGTH,
                                weights=[self.embedding_weights],
                                trainable=False
                                ))
        gru_model.add(GRU(64, return_sequences=True))
        gru_model.add(TimeDistributed(Dense(self.N_TOKENS, activation='softmax')))

        gru_model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['acc'])

        gru_model.summary()
        self.model = gru_model

    def __construct_LSTM_Model(self):
        lstm_model = Sequential()
        lstm_model.add(Embedding(input_dim=self.VOCABULARY_SIZE,  # vocabulary size - number of unique words in data
                                 output_dim=self.EMBEDDING_SIZE,  # length of vector with which each word is represented
                                 input_length=self.MAX_SEQ_LENGTH,  # length of input sequence
                                 weights=[self.embedding_weights],  # word embedding matrix
                                 trainable=False  # True - update embeddings_weight matrix
                                 ))
        lstm_model.add(LSTM(64, return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(self.N_TOKENS, activation='softmax')))

        lstm_model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['acc'])

        lstm_model.summary()
        self.model = lstm_model

    def buildModel(self, model_type):
        print("Model Building")
        if model_type not in self.__model_type:
            print("Type {} not supported.".format(model_type))
        elif model_type == self.__model_type[0]:
            self.__construct_GRU_model()
        elif model_type == self.__model_type[1]:
            self.__construct_bidirectional_LSTM_model()
        elif model_type == self.__model_type[2]:
            self.__construct_LSTM_Model()

    def fitModel(self):
        self.model.fit(self.train_X, self.train_Y,
                         batch_size=self.batch_size, epochs=self.epoch,
                         validation_data=(self.validation_X, self.validation_Y))
        # plt.plot(self.model.history('acc'))
        # plt.plot(self.model.history('val_acc'))
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc="lower right")
        # plt.show()

    def evaluateModel(self, test_X, test_Y):
        y = to_categorical(test_Y)
        loss, accuracy = self.model.evaluate(test_X, y, verbose=1)
        print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))
        result = self.model.predict(test_X)
        test_Y = np.ravel(test_Y)
        predict_result = np.ravel(np.argmax(result, axis=2))
        print("shape of result: ", result.shape)
        print("shape of predict_result: ", predict_result.shape)
        model_accuracy_score = accuracy_score(test_Y, predict_result)
        print("The accuracy of model: ", model_accuracy_score)
        model_f1_score = f1_score(test_Y, predict_result, labels=None, average='weighted', sample_weight=None)
        print("The F1 score of model: ", model_f1_score)
        model_recall_score = recall_score(test_Y, predict_result, labels=None, pos_label=1, average='weighted', sample_weight=None)
        print("The recall score of model: ", model_recall_score)
        print("Classification report of model:")
        print(classification_report(test_Y, predict_result, target_names=None, sample_weight=None))

