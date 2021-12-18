import numpy as np
import pickle
import os
from keras.preprocessing.text import Tokenizer


# read files word by word
def read_file_in_word(start, stop):
    documents = []
    tokens = []
    for i in range(start, stop + 1):
        file_name = "wsj_" + str(i).zfill(4) + ".dp"
        # print("File name : ", file_name)
        f = open("dependency_treebank/" + file_name)
        # print("file opened, file name : ", file_name)
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                # split line by " ", return a list of words, the first is document, second is token and third is feature
                single_data = line.split()
                documents.append(single_data[0])
                tokens.append(single_data[1])
        # print("file closed, file name : ", file_name)
        f.close()
    dataset = {"word": documents, "token": tokens}
    return dataset


# read files sentence by sentence with punctuations
def read_file_in_sentence(start, stop):
    sentences = []
    tokens = []
    single_documents = []
    single_tokens = []
    for i in range(start, stop + 1):
        sentences.append(single_documents)
        tokens.append(single_tokens)
        single_documents = []
        single_tokens = []
        file_name = "wsj_" + str(i).zfill(4) + ".dp"
        # print("File name : ", file_name)
        f = open("dependency_treebank/" + file_name)
        # print("file opened, file name : ", file_name)
        for line in f:
            if len(line) > 1:
                # split line by " ", return a list of words, the first is document, second is token and third is feature
                single_data = line.split()
                single_documents.append(single_data[0])
                single_tokens.append(single_data[1])
            else:
                sentences.append(single_documents)
                tokens.append(single_tokens)
                single_documents = []
                single_tokens = []
        # print("file closed, file name : ", file_name)
        f.close()
    sentences.remove([])
    tokens.remove([])
    dataset = {"sentences": sentences, "tokens": tokens}
    return dataset


# read whole file
total_word = read_file_in_word(1, 199)
total_sentence = read_file_in_sentence(1, 199)

total_X = total_sentence["sentences"]
total_Y = total_sentence["tokens"]
print("Sample of words: ", total_word["word"][0], "   ", total_word["token"][0])
print("Sample of words: ", total_word["word"][1], "   ", total_word["token"][1])
print("Sample of words: ", total_word["word"][2], "   ", total_word["token"][2])
print("Sample of sentences: ", total_X[0])
print("                     ", total_Y[0])
print("Sample of sentences: ", total_X[1], "   ")
print("                     ", total_Y[1])
print("Sample of sentences: ", total_X[2], "   ")
print("                     ", total_Y[2])

total_words_set = set(total_word["word"])
total_tokens_set = set(total_word["token"])
print("Total number of sentences: ", len(total_sentence["sentences"]))
print("Total number of words: ", len(total_words_set))
print("Total number of tokens: ", len(total_tokens_set))
# read file in train, validation and test datasets
train_sentence = read_file_in_sentence(1, 100)
train_words = set(read_file_in_word(1, 100)["word"])
train_X = train_sentence["sentences"]
train_Y = train_sentence["tokens"]


validation_sentence = read_file_in_sentence(101, 150)
validation_words = set(read_file_in_word(101, 150)["word"])
validation_X = validation_sentence["sentences"]
validation_Y = validation_sentence["tokens"]


test_sentence = read_file_in_sentence(151, 199)
test_words = set(read_file_in_word(151, 199)["word"])
test_X = test_sentence["sentences"]
test_Y = test_sentence["tokens"]


print("We have {} of train sentence.".format(len(train_X)))
print("We have {} of validation sentence.".format(len(validation_X)))
print("We have {} of test sentence.".format(len(test_X)))

pickled_data = {"total_words": total_words_set, "total_tokens": total_tokens_set, "total_X": total_X,
                "total_Y": total_Y, "train_X": train_X, "train_Y": train_Y, "validation_X": validation_X,
                "validation_Y": validation_Y, "test_X": test_X, "test_Y": test_Y, "train_words": train_words,
                "validation_words": validation_words, "test_words": test_words}

if not os.path.exists('PickledData/'):
    print('MAKING DIRECTORY PickledData/ to save pickled preprocessed data')
    os.makedirs('PickledData/')

with open('PickledData/pickled_data.pkl', 'wb') as f:
    pickle.dump(pickled_data, f)
