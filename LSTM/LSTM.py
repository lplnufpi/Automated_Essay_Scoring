from cProfile import label
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SpatialDropout1D, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, cohen_kappa_score, classification_report
import tensorflow as tf
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
    
    y_train_dummy = pd.get_dummies(y_train)
    y_test_dummy = pd.get_dummies(y_test)


    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # dot_img_file = 'model_1.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    epochs = 100
    batch_size = 10

    history = model.fit(X_train, y_train_dummy, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0001)])

   
    # # model.save('lstm_C4_model.h5')
    
    #model = load_model('lstm_C4_model.h5')
   
    accr = model.evaluate(X_test,y_test_dummy)
    print(accr)

    
    pred = model.predict(X_test)
    
    y_pred = []

    labels = [0, 40, 80, 120, 160, 200]
    for y in pred:
        y_pred.append(labels[np.argmax(y)])
    

    QWK = cohen_kappa_score(y_test, y_pred, weights = 'quadratic')

    print('Kappa: ', QWK)

    print(classification_report(y_test, y_pred, zero_division=0))