from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from matplotlib import pyplot as plt
N_TOKENS = 0
VOCABULARY_SIZE = 0
EMBEDDING_SIZE = 0
MAX_SEQ_LENGTH = 0
train_X = {}
train_Y = {}
validation_X = {}
validation_Y = {}
embedding_weights = []

# create architecture

bidirect_model = Sequential()
bidirect_model.add(Embedding(input_dim     = VOCABULARY_SIZE,
                             output_dim    = EMBEDDING_SIZE,
                             input_length  = MAX_SEQ_LENGTH,
                             weights       = [embedding_weights],
                             trainable     = True
))
bidirect_model.add(Bidirectional(LSTM(64, return_sequences=True)))
bidirect_model.add(TimeDistributed(Dense(N_TOKENS, activation='softmax')))

bidirect_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# check summary of model
bidirect_model.summary()

bidirect_training = bidirect_model.fit(train_X, train_Y, batch_size=128, epochs=10, validation_data=(validation_X, validation_Y))

# visualise training history
plt.plot(bidirect_training.history['acc'])
plt.plot(bidirect_training.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc="lower right")
plt.show()