from keras.preprocessing.text import Tokenizer
import pickle
import os
from keras.preprocessing.sequence import pad_sequences

# read preprocessed data
# pickled_data = {"total_words": total_words_set, "total_tokens": total_tokens_set, "total_X": total_X,
#                 "total_Y": total_Y, "train_X": train_X, "train_Y": train_Y, "validation_X": validation_X,
#                 "validation_Y": validation_Y, "test_X": test_X, "test_Y": test_Y}
# total_words: total words set
# total_tokens: total tokens set
# total_X: total sentences in data
# total_Y: total tokens of sentences in data, total_Y and total_X are pairs
#
# train_X: train sentences in data
# train_Y: train tokens of sentences in data, train_Y and train_X are pairs
#
# validation_X: validation sentences in data
# validation_Y: validation tokens of sentences in data, validation_Y and validation_X are pairs
#
# test_X: test sentences in data
# test_Y: test tokens of sentences in data, test_Y and test_X are pairs
with open('PickledData/pickled_data.pkl', 'rb') as f:
    pickled_data = pickle.load(f)
    total_words = pickled_data["total_words"]
    total_tokens = pickled_data["total_tokens"]

    total_X = pickled_data["total_X"]
    total_Y = pickled_data["total_Y"]

    train_X = pickled_data["train_X"]
    train_Y = pickled_data["train_Y"]

    validation_X = pickled_data["validation_X"]
    validation_Y = pickled_data["validation_Y"]

    test_X = pickled_data["test_X"]
    test_Y = pickled_data["test_Y"]

# Vectorise X and Y
# Encode X and Y to integer values
word2int = {}
int2word = {}
token2int = {}
int2token = {}
# generate dict from word to index, index to word, token to index and index to token
for i, word in enumerate(total_words):
    word2int[word] = i + 1
    int2word[i + 1] = word

for i, token in enumerate(total_tokens):
    token2int[token] = i + 1
    int2token[i + 1] = token


def tokenizer(sequences, dicts):
    encoded_sequences = []
    for sub_sequence in sequences:
        encoded_sub_sequence = []
        for item in sub_sequence:
            encoded_sub_sequence.append(dicts[item])
        encoded_sequences.append(encoded_sub_sequence)
    return encoded_sequences


encoded_train_X = tokenizer(train_X, word2int)
encoded_train_Y = tokenizer(train_Y, token2int)

encoded_validation_X = tokenizer(validation_X, word2int)
encoded_validation_Y = tokenizer(validation_Y, token2int)

encoded_test_X = tokenizer(test_X, word2int)
encoded_test_Y = tokenizer(test_Y, token2int)


print("Train sample \n", "-" * 50, "\n")
print('X: ', train_X[0], '\n')
print('Y: ', train_Y[0], '\n')
print()
print("Encoded train sample \n", "-" * 50, "\n")
print('X: ', encoded_train_X[0], '\n')
print('Y: ', encoded_train_Y[0], '\n')
print("-" * 50, "\n")
print("Validation sample \n", "-" * 50, "\n")
print('X: ', validation_X[0], '\n')
print('Y: ', validation_Y[0], '\n')
print()
print("Encoded validation sample \n", "-" * 50, "\n")
print('X: ', encoded_validation_X[0], '\n')
print('Y: ', encoded_validation_Y[0], '\n')
print("-" * 50, "\n")
print("Test sample \n", "-" * 50, "\n")
print('X: ', test_X[0], '\n')
print('Y: ', test_Y[0], '\n')
print()
print("Encoded test sample \n", "-" * 50, "\n")
print('X: ', encoded_test_X[0], '\n')
print('Y: ', encoded_test_Y[0], '\n')
print("-" * 50, "\n")

# length check, each pair of input and output sequence should have same length

different_length = [1 if len(input) != len(output) else 0 for input, output in zip(encoded_train_X, encoded_train_Y)]
print("{} sentences have disparate input-output lengths in train samples.".format(sum(different_length)))

different_length = [1 if len(input) != len(output) else 0 for input, output in
                    zip(encoded_validation_X, encoded_validation_Y)]
print("{} sentences have disparate input-output lengths in validation samples.".format(sum(different_length)))

different_length = [1 if len(input) != len(output) else 0 for input, output in zip(encoded_test_X, encoded_test_Y)]
print("{} sentences have disparate input-output lengths in test samples.".format(sum(different_length)))

# pad sequence
# As of now, the sentences present in the data are of various lengths.
# We need to either pad short sentences or truncate long sentences to a fixed length.
# This fixed length is a hyperparameter.

# Pad each sequence to MAX_SEQ_LENGTH using KERAS' pad_sequences() function.
# Sentences longer than MAX_SEQ_LENGTH are truncated.
# Sentences shorter than MAX_SEQ_LENGTH are padded with zeroes.

# Truncation and padding can either be 'pre' or 'post'.
# For padding we are using 'pre' padding type, that is, add zeroes on the left side.
# For truncation, we are using 'post', that is, truncate a sentence from right side.

MAX_SEQ_LENGTH = 100
train_X = padded_train_X = pad_sequences(encoded_train_X, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
train_Y = padded_train_Y = pad_sequences(encoded_train_Y, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

validation_X = padded_validation_X = pad_sequences(encoded_validation_X, maxlen=MAX_SEQ_LENGTH, padding="pre",
                                                   truncating="post")
validation_Y = padded_validation_Y = pad_sequences(encoded_validation_Y, maxlen=MAX_SEQ_LENGTH, padding="pre",
                                                   truncating="post")

test_X = padded_test_X = pad_sequences(encoded_test_X, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
test_Y = padded_test_Y = pad_sequences(encoded_test_Y, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

padded_samples = {"train_X": train_X, "train_Y": train_Y, "validation_X": validation_X, "validation_Y": validation_Y,
                  "test_X": test_X, "test_Y": test_Y, "MAX_SEQ_LENGTH": 100, "int2token": int2token,
                  "int2word": int2word}

if not os.path.exists('PaddedData/'):
    print('MAKING DIRECTORY PaddedData/ to save pickled padded samples')
    os.makedirs('PaddedData/')

with open('PaddedData/padded_samples.pkl', 'wb') as f:
    pickle.dump(padded_samples, f)
