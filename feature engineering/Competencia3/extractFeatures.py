import pandas as pd
from FeatsExtractor import FeatsExtractor
import pickle


if __name__ == '__main__':
     ft = FeatsExtractor()

     target = 'c3'

     corpus_filepath = target+'_train90.csv'
     df_train = pd.read_csv(corpus_filepath, converters={'essay': eval})

     corpus_filepath = target+'_test10.csv'
     df_test = pd.read_csv(corpus_filepath, converters={'essay': eval})

     
     print('\n')

     #Extraindo faetures e gravando em arquivo 

     print("Corpus de treino: ")
     X_train,y_train = ft.extractCorpus(df_train)
     with open('X_train.pickle', 'wb') as f:
          pickle.dump(X_train, f)
     f.close()

     with open('y_train.pickle', 'wb') as f:
          pickle.dump(y_train, f)
     f.close()


     print('\n')
     
     print("Corpus de teste: ")
     X_test,y_test = ft.extractCorpus(df_test)
     with open('X_test.pickle', 'wb') as f:
          pickle.dump(X_test, f)
     f.close()

     with open('y_test.pickle', 'wb') as f:
          pickle.dump(y_test, f)
     f.close()
     
     print('\n')
