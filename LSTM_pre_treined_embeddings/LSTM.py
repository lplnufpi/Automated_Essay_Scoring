import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import text, sequence 
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers import TextVectorization
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import cohen_kappa_score, classification_report
import tensorflow as tf
import sys
from data import CorpusRedacao
import statistics


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    data = CorpusRedacao()
    X_train, y_train, X_test, y_test = data.loadSplits('c4')  

    max_len = int(statistics.mean(len(x) for x in X_train ))
    print(f'Max number of words in a text in training data: {max_len}')

    max_words = 1000
    tokenizer = text.Tokenizer(num_words = max_words)
    # create the vocabulary by fitting on x_train text
    tokenizer.fit_on_texts(X_train)
    # generate the sequence of tokens
    xtrain_seq = tokenizer.texts_to_sequences(X_train)
    xtest_seq = tokenizer.texts_to_sequences(X_test)

    # pad the sequences
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)
    word_index = tokenizer.word_index

 
    embedding_vectors = {}
    with open('embeddings/cbow_s300.txt','r',encoding='utf-8') as file:
        for row in file:
            values = row.split(' ')
            word = values[0]
            weights = np.asarray([float(val) for val in values[1:]])
            embedding_vectors[word] = weights
    print(f"Size of vocabulary in CBOW: {len(embedding_vectors)}")


    #initialize the embedding_matrix with zeros
    emb_dim = 300
    if max_words is not None: 
        vocab_len = max_words 
    else:
        vocab_len = len(word_index)+1
    embedding_matrix = np.zeros((vocab_len, emb_dim))
    oov_count = 0
    oov_words = []
    for word, idx in word_index.items():
        if idx < vocab_len:
            embedding_vector = embedding_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
            else:
                oov_count += 1 
                oov_words.append(word)
    #print some of the out of vocabulary words
    print('Some out of valubulary words:', len(oov_words))
    print('Size of embedding matrix ', embedding_matrix.shape)

    lstm_model = Sequential()
    lstm_model.add(Embedding(vocab_len, emb_dim, trainable = False, weights=[embedding_matrix]))
    lstm_model.add(SpatialDropout1D(0.5))
    lstm_model.add(LSTM(10, dropout=0.5, recurrent_dropout=0.5))
    lstm_model.add(Dense(6))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(lstm_model.summary())

    y_train_dummy = pd.get_dummies(y_train)
    y_test_dummy = pd.get_dummies(y_test)

    epochs = 10
    batch_size = 100

    history = lstm_model.fit(xtrain_pad, y_train_dummy, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

   
    lstm_model.save('Embedding_lstm_C4_model.h5')
   
    #lstm_model = load_model('Embedding_lstm_C5_model.h5')
   
    accr = lstm_model.evaluate(xtest_pad,y_test_dummy)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    pred = lstm_model.predict(xtest_pad)
    
    y_pred = []

    labels = [0, 40, 80, 120, 160, 200]
    for y in pred:
        y_pred.append(labels[np.argmax(y)])
    

    QWK = cohen_kappa_score(y_test, y_pred, weights = 'quadratic')

    print('Kappa:', QWK)

    print(classification_report(y_test, y_pred, zero_division=0))