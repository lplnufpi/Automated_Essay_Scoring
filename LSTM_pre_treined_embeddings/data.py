import re
import pandas as pd
from nltk import corpus
import re
from sklearn.model_selection import train_test_split

class CorpusRedacao:

    def __init__(self):
        self.REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        self.STOPWORDS = set(corpus.stopwords.words("portuguese"))


    def clean_text(self, text):
        text = text.lower() # lowercase text
        text = self.REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = self.BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS) # remove stopwors from text
        
        return text


    def loadSplits(self,target = 'score'):
        corpus_filepath = target+'_X_train90.csv'
        df_train = pd.read_csv(corpus_filepath, converters={'essay': eval})

        corpus_filepath = target+'_X_test10.csv'
        df_test = pd.read_csv(corpus_filepath, converters={'essay': eval})

        corpus_filepath = target+'_y_train90.csv'
        y_train = pd.read_csv(corpus_filepath, converters={'essay': eval})

        corpus_filepath = target+'_y_test10.csv'
        y_test = pd.read_csv(corpus_filepath, converters={'essay': eval})

        X_train = []
        for x in df_train.itertuples():
            texto = '\n'.join(x.essay)
            texto = self.clean_text(texto)
            X_train.append(texto)
        
        X_test = []
        for x in df_test.itertuples():
            texto = '\n'.join(x.essay)
            texto = self.clean_text(texto)
            X_test.append(texto)
        
        return X_train, y_train.squeeze().to_numpy(), X_test, y_test.squeeze().to_numpy()



    def loadCorpus(self, target = 'score'):
        corpus_filepath = 'new_essay-br.csv'
        df = pd.read_csv(corpus_filepath, converters={'essay': eval})

        X = []
        for x in df.itertuples():
            texto = '\n'.join(x.essay)
            texto = self.clean_text(texto)
            X.append(texto)
        
        return X, df[target].squeeze().to_numpy()
        

if __name__ == '__main__':
    cp = CorpusRedacao()

    X,y = cp.loadCorpus('c5')

    print(X)