# POSTagging

# Introduce

This is a simple POS Tagging sample. 

Implemented net word architecture:

1. A Bidirectional LSTM layer and a Dense/Fully-Connected layer on top
2. Using a GRU instead of the LSTM
3. Adding an additional LSTM layer
4. Adding an additional dense layer

# Data

With data set given: dependency_treebank

# Preprocess Data

In file PreprocessData.py, files were read by sentence and word. After reading, pickled all the data and store them into SSD as pickled_data.pkl.
In this step, we store all of the data, and do some simple classification by ourself, which including:

  1. Get total word set
  2. Get total token set
  3. Get train data X
  4. Get train token Y
  5. Get validation data X
  6. Get validation token Y
  7. Get test data X
  8. Get test token Y
  9. Replace "&" to ",", since we find in training token set there's a "&" but in validation and test token set there's not.
  
 # Preprocess GloVe file
 
In file PickleGloVe.py, we pickled GloVe dictionary and store it into SSD. In this way, we can read and get GloVe data faster.
GloVe chosen: glove-wiki-gigaword-300.txt
 
 # Paddomh Data
 
In file PaddingSample.py, all data needed was paded and stored into SSD.

Automatically padding length is 100. We first tokenized dictionaries called Total_words abd Total_tokens, and constructed 4 maps, there're:

  1. word2int: e.g word2int["joy"] = 1
  2. int2word: e.g int2word[1] = "joy"
  3. token2int: e.g token2int["NN"] = 2
  4. int2token: e.g int2token[2] = "NN"


After we have these 4 maps, we can easily tokenize all of the data we need. Data we need are:
  1. encoded_train_X: smaples for training
  2. encoded_train_Y: tags for training
  3. encoded_validation_X: smaples for validation
  4. encoded_validation_Y: tags for validation
  5. encoded_test_X: smaples for test
  6. encoded_test_Y: tags for test


With all of these data, we store them into SSD as padded_samples: 

padded_samples = {"train_X": train_X, "train_Y": train_Y, "validation_X": validation_X, "validation_Y": validation_Y,
                  "test_X": test_X, "test_Y": test_Y, "MAX_SEQ_LENGTH": 100, "int2token": int2token,
                  "int2word": int2word}
                  
                  
  # Embedding 
We embedding with GloVe in 300 dimension.

Read all the dictionary we need from SSD, there are data we stored during preprocesing data:

  1. train_words
  2. validation_words
  3. test_words


Following intro:

  1. We have V1 = GloVe dictionary
  2. Find words exists in train_words but not in V1, which we call it OOV1.
  3. Have a new dctionary V2 = OOV1 + part of V1(those words exsits both in train_words and V1)
  4. Find words exists in validation_words but not in V2, which we call it OOV2.
  5. Have a new dctionary V3 = OOV2 + part of V2(those words exsits both in train_words and V2)
  6. Find words exists in validation_words but not in V3, which we call it OOV3.
  7. Have a new dctionary V4 = OOV3 + part of V3(those words exsits both in train_words and V3)
  8. After get the dictionary, we turn them into matrixs to get embedded weights


Steps in embedding look a little bit strange. Howecer, we given OOV words vectors with uniform random array by code np.random.uniform(size=(1, EMBEDDING_DIMENSION))


# Net work architecture

In file RnnModel.py, we can built several network:

  1. Bidirectional_LSTM_Model
  2. Gru_Model
  3. LSTM_Model
  4. Bidirectional_LSTM_Model with 2 dense layer


New a instance of POSTaggingModel, and input general parameters.

  POSTaggingModel:
    N_TOKENS, 
    VOCABULARY_SIZE,
    EMBEDDING_SIZE, 
    MAX_SEQ_LENGTH, 
    train_X, 
    train_Y,
    validation_X, 
    validation_Y, 
    embedding_weights, 
    batch_size, 
    epoch, 
    name
    
 Use POSTaggingModel.buildModel(type) to build a model.
    type is a private list fixed as: 
    ["GRU Model", "Bidirectional LSTM Model", "LSTM Model", "Bidirectional LSTM Model with 2 Dense", "2 LSTM Model"]
    Only type in this list can build model.
    
    
 In POSTaggingModel.buildModel(type), it will compile model and print the summary for model.
 
 After this, use model.fit() to train model.
 
 After train, use model.evaluateModel() to test model and save report.
