import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    corpus = pd.read_csv('essay-br.csv', converters={'essay': eval})   
    df_remove = corpus.loc[(corpus['score'] == 0)]
    corpus = corpus.drop(df_remove.index)
    y = corpus.c4
    X_train, X_test, y_train, y_test = train_test_split(corpus, y,train_size=0.9,test_size=0.1, random_state=None)

    X_train.to_csv('C4_X_train90.csv', index=False)
    X_test.to_csv('C4_X_test10.csv', index=False)
    y_train.to_csv('C4_y_train90.csv', index=False)
    y_test.to_csv('C4_y_test10.csv', index=False)