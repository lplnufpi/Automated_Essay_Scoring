from cProfile import label
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout, SpatialDropout1D, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, cohen_kappa_score
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from data import CorpusRedacao

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 30000
# Max number of words in each essay.
MAX_SEQUENCE_LENGTH = 1000
# This is fixed.
EMBEDDING_DIM = 300

def discretize(y_pred):
        y_cat = []
        for t in y_pred:
            if t[0] < 20:
                y_cat.append(0)
            elif 20 <= t[0] < 60:
                y_cat.append(40)
            elif 60 <= t[0] < 100:
                y_cat.append(80)
            elif 100 <= t[0] < 140:
                y_cat.append(120)
            elif 140 <= t[0] < 180:
                y_cat.append(160)
            elif 180 <= t[0]:
                y_cat.append(200)

        return y_cat


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    data = CorpusRedacao()
    X_train, y_train, X_test, y_test = data.load('c4')

    
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    

    i = 0
    for y in y_train:
        if y == 40:
            y_train[i] = 1
        elif y == 80:
            y_train[i] = 2
        elif y == 120:
            y_train[i] = 3
        elif y == 160:
            y_train[i] = 4
        elif y == 200:
            y_train[i] = 5
        i += 1

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(10, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(1))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='sgd', metrics=[tfa.metrics.CohenKappa(num_classes=6,sparse_labels=True,regression=True, weightage='quadratic')])

    print(model.summary())

    epochs = 100
    batch_size = 10

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, restore_best_weights=True )])

   
    model.save('regr_lstm_model.h5')
    
    # model = load_model('regr_lstm_C5_model.h5')
   
    accr = model.evaluate(X_test,y_test)
    print(accr)

    
    pred = model.predict(X_test)

    print(pred)

