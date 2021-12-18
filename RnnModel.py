from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from matplotlib import pyplot as plt


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
                                     trainable=True
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
                                trainable=True
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
                                 trainable=True  # True - update embeddings_weight matrix
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
        plt.plot(self.__model.history['acc'])
        plt.plot(self.__model.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc="lower right")
        plt.show()

    def evaluateModel(self, test_X, test_Y):
        loss, accuracy = self.model.evaluate(test_X, test_Y, verbose=1)
        print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))


