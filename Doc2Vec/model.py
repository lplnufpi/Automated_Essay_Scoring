import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error, classification_report
from nltk import word_tokenize, corpus
import gensim
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

class ModelDoc2Vec:
    def load_corpus(self,target = 'score'):
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
            X_train.append(texto)
        
        X_test = []
        for x in df_test.itertuples():
            texto = '\n'.join(x.essay)
            X_test.append(texto)
        
        return X_train, y_train, X_test, y_test




    def prepare_corpus(self, redacoes):
        stoplist = corpus.stopwords.words("portuguese")
        tokens = []
        for i, redacao in enumerate(redacoes):
            tokens = [p.lower() for p in word_tokenize(redacao, language='portuguese') if p.isalpha() and p not in stoplist]
            
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])




    def train_and_save(self, corpus_filepath):
        X, y = self.load_corpus(corpus_filepath) 
        documents = list(self.prepare_corpus(X))
        model = gensim.models.Doc2Vec(dm = 1, vector_size=300, window=4, min_count=1, epochs=100, sample=1e-4, workers=5)#PV-DM
        #model = gensim.models.Doc2Vec(dm = 2, vector_size=300, window=4, min_count=1, epochs=100, sample=1e-4, workers=5)#PV-DBOW
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save('saved_models/doc2vecPVDM.model')



    def get_model(self, filepath):
        model = gensim.models.Doc2Vec.load(filepath)
        
        return model



    def split_data_set(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, test_size = 0.1, random_state = 42)

        return X_train, X_test, y_train, y_test




    def train_doc2vec_embeddings(self, vec_size, option, data):
        documents = list(self.prepare_corpus(data))
        model = gensim.models.Doc2Vec(vector_size = vec_size, dm = option, window=4, min_count=1, epochs=300, sample=1e-4, workers=5)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    
        return model

    


    def infer_d2v_embeddings(self, d2v_model, X):
        train_data = []
        for i in range(len(X)):
            model_vector = d2v_model.infer_vector(gensim.utils.simple_preprocess(X[i]))
            train_data.append(model_vector)
            
        return train_data




    def discretize(self, y_pred):
        y_cat = []
        for t in y_pred:
            if t < 35:
                y_cat.append(0)
            elif 35 <= t < 60:
                y_cat.append(40)
            elif 60 <= t < 100:
                y_cat.append(80)
            elif 100 <= t < 140:
                y_cat.append(120)
            elif 140 <= t < 180:
                y_cat.append(160)
            elif 180 <= t:
                y_cat.append(200)

        return y_cat




    def run_model(self, classifier, X_train, X_test, y_train, y_test):
        
        classifier.fit(X_train,y_train)
        
        pred = classifier.predict(X_test)        
        pred = self.discretize(pred)

        QWK = cohen_kappa_score(y_test, pred, weights = 'quadratic')
        MSE = mean_squared_error(y_test, pred, squared=False)
        classif_report = classification_report(y_test, pred, zero_division=0)
        
        return QWK, MSE, classif_report
