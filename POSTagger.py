# @Time : 2021/12/16 下午8:25 
# @Author : Patrick.Lai
# @File : POSTagger.py 
# @Software: PyCharm
from ExtractData import Corpus
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
with open('PickledData/total_data.pkl', 'rb') as f:
    total_data = pickle.load(f)
    total_word = total_data.word2int
    total_token = total_data.token2int
    total_sentences = total_data.sentences2int
    total_tokens = total_data.tokens2int

test_sentence = "I am a boy".split()

numerized_sentence = []

for word in test_sentence:
	numerized_sentence.append(total_word[word])

numerized_sentence = np.asarray([numerized_sentence])
padded_numerized_sentence = pad_sequences(numerized_sentence, maxlen=150)

print('The sentence is ', test_sentence)
print('The tokenized sentence is ',numerized_sentence)
print('The padded tokenized sentence is ', padded_numerized_sentence)

model = load_model('Models/model.h5')

prediction = model.predict(padded_numerized_sentence)

print(prediction.shape)

handled_prediction = np.argmax(prediction, axis = 2)

print(handled_prediction)

