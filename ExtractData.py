# @Time : 2021/12/6 下午3:09 
# @Author : Patrick.Lai
# @File : ExtractData.py
# @Software: PyCharm
import numpy as np
import pickle
import os


def read_file_in_word(start, stop):
    documents = []
    tokens = []
    for i in range(start, stop + 1):
        file_name = "wsj_" + str(i).zfill(4) + ".dp"
        print("File name : ", file_name)
        f = open("dependency_treebank/" + file_name)
        print("file opened, file name : ", file_name)
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                # split line by " ", return a list of words, the first is document, second is token and third is feature
                single_data = line.split()
                documents.append(single_data[0])
                tokens.append(single_data[1])
        print("file closed, file name : ", file_name)
        f.close()
    dataset = {"document": documents, "token": tokens}
    return dataset


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
        print("File name : ", file_name)
        f = open("dependency_treebank/" + file_name)
        print("file opened, file name : ", file_name)
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
        print("file closed, file name : ", file_name)
        f.close()
    sentences.remove([])
    tokens.remove([])
    dataset = {"sentences": sentences, "tokens": tokens}
    return dataset


# read whole data

whole_data_in_word = read_file_in_word(1, 199)
whole_data_in_sentence_with_punctuation = read_file_in_sentence(1, 199)

total_tokens_in_word = whole_data_in_word["token"]
total_words = whole_data_in_word["document"]

total_words_set = set(total_words)
total_tokens_set = set(total_tokens_in_word)

# numerize total data

word2int = {"train": {}, "validation": {}, "test": {}, "global": {}}
int2word = {"train": {}, "validation": {}, "test": {}, "global": {}}
token2int = {"train": {}, "validation": {}, "test": {}, "global": {}}
int2token = {"train": {}, "validation": {}, "test": {}, "global": {}}

for i, word in enumerate(total_words_set):
    word2int["global"][word] = i + 1
    int2word["global"][i + 1] = word

for i, word in enumerate(total_tokens_set):
    token2int["global"][word] = i + 1
    int2token["global"][i + 1] = word

# read training, validation and test data
train_data_in_word = read_file_in_word(1, 100)
validation_data_in_word = read_file_in_word(101, 150)
test_data_in_word = read_file_in_word(151, 199)
train_data_in_sentence_with_punctuation = read_file_in_sentence(1, 100)
validation_data_in_sentence_with_punctuation = read_file_in_sentence(101, 150)
test_data_in_sentence_with_punctuation = read_file_in_sentence(151, 199)



train_tokens_in_word = train_data_in_word["token"]
train_words = train_data_in_word["document"]
validation_tokens_in_word = validation_data_in_word["token"]
validation_words = validation_data_in_word["document"]
test_tokens_in_word = test_data_in_word["token"]
test_words = test_data_in_word["document"]

train_words_set = set(train_words)
train_tokens_set = set(train_tokens_in_word)
validation_words_set = set(validation_words)
validation_tokens_set = set(validation_tokens_in_word)
test_words_set = set(test_words)
test_tokens_set = set(test_tokens_in_word)

print("Total number of words in training set: ", len(train_words_set))
print("Total number of tokens in training set: ", len(train_tokens_set))
print("Total number of words in validation set: ", len(validation_words_set))
print("Total number of tokens in validation set: ", len(validation_tokens_set))
print("Total number of words in test set: ", len(test_words_set))
print("Total number of tokens in test set: ", len(test_tokens_set))



for i, word in enumerate(train_words_set):
    word2int["train"][word] = i + 1
    int2word["train"][i + 1] = word

for i, word in enumerate(train_tokens_set):
    token2int["train"][word] = i + 1
    int2token["train"][i + 1] = word

for i, word in enumerate(validation_words_set):
    word2int["validation"][word] = i + 1
    int2word["validation"][i + 1] = word

for i, token in enumerate(validation_tokens_set):
    token2int["validation"][token] = i + 1
    int2token["validation"][i + 1] = token

for i, token in enumerate(test_words_set):
    word2int["test"][token] = i + 1
    int2word["test"][i + 1] = token

for i, token in enumerate(test_tokens_set):
    token2int["test"][token] = i + 1
    int2token["test"][i + 1] = token

# train_data_in_sentence_with_punctuation
# validation_data_in_sentence_with_punctuation
# test_data_in_sentence_with_punctuation

sentence2int = {"train": [], "validation": [], "test": []}
tokens2int = {"train": [], "validation": [], "test": []}

train_sentences = train_data_in_sentence_with_punctuation["sentences"]
train_sentences_tokens = train_data_in_sentence_with_punctuation["tokens"]

for sentence in train_sentences:
    numrized_sentence = []
    words = word2int["train"]
    for word in sentence:
        numrized_sentence.append(words[word])
    sentence2int["train"].append(numrized_sentence)

for setences_tokens in train_sentences_tokens:
    numrized_tokens = []
    tokens = token2int["train"]
    for token in setences_tokens:
        numrized_tokens.append(tokens[token])
    tokens2int["train"].append(numrized_tokens)

sentence2int["train"] = np.array(sentence2int["train"])
tokens2int["train"] = np.asarray(tokens2int["train"])

validation_sentences = validation_data_in_sentence_with_punctuation["sentences"]
validation_sentences_tokens = validation_data_in_sentence_with_punctuation["tokens"]

for sentence in validation_sentences:
    numrized_sentence = []
    words = word2int["validation"]
    for word in sentence:
        numrized_sentence.append(words[word])
    sentence2int["validation"].append(numrized_sentence)

for setences_tokens in validation_sentences_tokens:
    numrized_tokens = []
    tokens = token2int["validation"]
    for token in setences_tokens:
        numrized_tokens.append(tokens[token])
    tokens2int["validation"].append(numrized_tokens)

sentence2int["validation"] = np.array(sentence2int["validation"])
tokens2int["validation"] = np.asarray(tokens2int["validation"])

test_sentences = test_data_in_sentence_with_punctuation["sentences"]
test_sentences_tokens = test_data_in_sentence_with_punctuation["tokens"]

for sentence in test_sentences:
    numrized_sentence = []
    words = word2int["test"]
    for word in sentence:
        numrized_sentence.append(words[word])
    sentence2int["test"].append(numrized_sentence)

for setences_tokens in test_sentences_tokens:
    numrized_tokens = []
    tokens = token2int["test"]
    for token in setences_tokens:
        numrized_tokens.append(tokens[token])
    tokens2int["test"].append(numrized_tokens)

sentence2int["test"] = np.array(sentence2int["test"])
tokens2int["test"] = np.asarray(tokens2int["test"])


class Corpus:
    def __init__(self, word2int, int2word, token2int, int2token, sentence2int, tokens2int):
        self.word2int = word2int
        self.int2word = int2word
        self.token2int = token2int
        self.int2token = int2token
        self.sentences2int = sentence2int
        self.tokens2int = tokens2int


train_corpus = Corpus(word2int["train"], int2word["train"], token2int["train"], int2token["train"],
                      sentence2int["train"], tokens2int["train"])
validation_corpus = Corpus(word2int["validation"], int2word["validation"], token2int["validation"],
                           int2token["validation"], sentence2int["validation"], tokens2int["validation"])
test_corpus = Corpus(word2int["test"], int2word["test"], token2int["test"], int2token["test"], sentence2int["test"],
                     tokens2int["test"])
total_corpus = Corpus(word2int["global"], int2word["global"], token2int["global"], int2token["global"], {},
                      {})

if not os.path.exists('PickledData/'):
    print('MAKING DIRECTORY PickledData/ to save pickled glove file')
    os.makedirs('PickledData/')

with open('PickledData/train_data.pkl', 'wb') as f:
    pickle.dump(train_corpus, f)

with open('PickledData/validation_data.pkl', 'wb') as f:
    pickle.dump(validation_corpus, f)

with open('PickledData/test_data.pkl', 'wb') as f:
    pickle.dump(test_corpus, f)

print(1)
