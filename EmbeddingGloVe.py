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


def get_embedding_weights(vocabulary, size):
    embedding_weights = []
    for key in vocabulary:
        embedding_weights.append(vocabulary[key])
    return np.array(embedding_weights)


print("Original length of GloVe vocabulary: {} \n".format(len(V1)))
OOV1_counters = 1
OOV1 = {}
for word in train_words:
    if word not in V1.keys():
        OOV1_counters += 1
        OOV1[word] = np.random.uniform(size=(1, EMBEDDING_DIMENSION))
print("Find {} words in OOV1".format(OOV1_counters))
V2 = {**V1, **OOV1}
embedding_weights_V2 = get_embedding_weights(V2)
print("V2 = V1 + OOV1, shape of V2: {} \n".format(embedding_weights_V2.shape))

OOV2_counters = 1
OOV2 = {}
for word in validation_words:
    if word not in V2.keys():
        OOV2_counters += 1
        OOV2[word] = np.random.uniform(size=(1, EMBEDDING_DIMENSION))
print("Find {} words in OOV2".format(OOV2_counters))
V3 = {**V2, **OOV2}
embedding_weights_V3 = np.asarray(V3.values())
print("V3 = V1 + OOV1 + OOV2, shape of V3: {} \n".format(embedding_weights_V3.shape))

OOV3_counters = 1
OOV3 = {}
for word in test_words:
    if word not in V3.keys():
        OOV3_counters += 1
        OOV3[word] = np.random.uniform(size=(1, EMBEDDING_DIMENSION))
print("Find {} words in OOV3".format(OOV3_counters))
V4 = {**V3, **OOV3}
embedding_weights_V4 = np.asarray(V4.values())
print("V4 = V1 + OOV1 + OOV2 + OOV3, shape of V4: {} \n".format(embedding_weights_V4.shape))
