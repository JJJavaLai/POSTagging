import pickle
import os
import numpy as np

with open('PickledData/Glove.pkl', 'rb') as f:
    V1 = pickle.load(f)

print('Total %s word vectors. \n' % len(V1))

# train_words: train word set,
# validation_words: validation words set,
# test_words: test words set

with open('PickledData/pickled_data.pkl', 'rb') as f:
    pickled_data = pickle.load(f)
    train_words = pickled_data["train_words"]
    validation_words = pickled_data["validation_words"]
    test_words = pickled_data["test_words"]

EMBEDDING_DIMENSION = 300


def get_embedding_weights(vocabulary):
    shape = (len(vocabulary), EMBEDDING_DIMENSION)
    embedding_weights = np.empty((shape))
    keys = list(vocabulary.keys())
    for i in range(0, len(keys)):
        embedding_weights[i] = vocabulary[keys[i]]
    return embedding_weights


print("Original length of GloVe vocabulary: {} \n".format(len(V1)))
OOV1_counters = 1
train_vocabulary_size = len(train_words)
V2 = {}
for word in train_words:
    if word not in V1.keys():
        OOV1_counters += 1
        V2[word] = np.random.uniform(size=(1, EMBEDDING_DIMENSION))
    else:
        V2[word] = V1[word]
print("Find {} words in OOV1".format(OOV1_counters))
embedding_weights_V2 = get_embedding_weights(V2)
print("V2 = V1 + OOV1, shape of V2: {} \n".format(embedding_weights_V2.shape))

OOV2_counters = 1
V3 = {}
for word in validation_words:
    if word not in V2.keys():
        OOV2_counters += 1
        V3[word] = np.random.uniform(size=(1, EMBEDDING_DIMENSION))
    else:
        V3[word] = V2[word]
print("Find {} words in OOV2".format(OOV2_counters))
embedding_weights_V3 = get_embedding_weights(V3)
print("V3 = V1 + OOV1 + OOV2, shape of V3: {} \n".format(embedding_weights_V3.shape))

OOV3_counters = 1
V4 = {}
for word in test_words:
    if word not in V3.keys():
        OOV3_counters += 1
        V4[word] = np.random.uniform(size=(1, EMBEDDING_DIMENSION))
    else:
        V4[word] = V3[word]
print("Find {} words in OOV3".format(OOV3_counters))
embedding_weights_V4 = get_embedding_weights(V4)
print("V4 = V1 + OOV1 + OOV2 + OOV3, shape of V4: {} \n".format(embedding_weights_V4.shape))

if not os.path.exists('PickledData/'):
    print('MAKING DIRECTORY PickledData/ to save pickled preprocessed data')
    os.makedirs('PickledData/')

with open('PickledData/V2.pkl', 'wb') as f:
    pickle.dump(embedding_weights_V2, f)

with open('PickledData/V3.pkl', 'wb') as f:
    pickle.dump(embedding_weights_V3, f)

with open('PickledData/V4.pkl', 'wb') as f:
    pickle.dump(embedding_weights_V4, f)
