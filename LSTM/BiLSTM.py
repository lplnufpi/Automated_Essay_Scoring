from  tensorflow  import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from data import CorpusRedacao
from sklearn.metrics import cohen_kappa_score
import sys


MAX_NB_WORDS = 3000  # Only consider the top 1500 words
MAX_SEQUENCE_LENGTH = 350  # Only consider the first 1000 words of essay

if __name__ == '__main__':


    np.set_printoptions(threshold=sys.maxsize)

    data = CorpusRedacao()
    X_train, y_train, X_test, y_test = data.load('c5')

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    print(y_test)
    
    y_train_dummy = pd.get_dummies(y_train)
    y_test_dummy = pd.get_dummies(y_test)

    print(y_test_dummy)


    inputs = keras.Input(shape=(None,))
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(MAX_NB_WORDS, 300)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(6, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    epochs = 100
    batch_size = 100

    history = model.fit(X_train, y_train_dummy, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

   
    model.save('Bilstm_C5_model.h5')
   
   
    accr = model.evaluate(X_test,y_test_dummy)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    #model = load_model('lstm_C4_model.h5')

    pred = model.predict(X_test)
    
    y_pred = []

    labels = [0, 40, 80, 120, 160, 200]
    for y in pred:
        y_pred.append(labels[np.argmax(y)])
    

    QWK = cohen_kappa_score(y_test, y_pred, weights = 'quadratic')

    print('Kappa: ', QWK)