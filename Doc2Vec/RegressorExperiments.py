from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import tree  
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import r2_score, cohen_kappa_score, mean_squared_error, f1_score, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import normalize, Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

import pickle

from tqdm import tqdm

from model import ModelDoc2Vec

# from imblearn.over_sampling import BorderlineSMOTE

import pandas as pd



if __name__ == '__main__':
    model = ModelDoc2Vec()

    vec_size = 50
    model_option = 0

    redacoes = 'new_essay-br.csv'

    print('Carregando corpus...\n')

    target = 'c1'

    X_train, y_train, X_test, y_test = model.load_corpus(target)

    y_train = y_train[target].tolist()

    y_test = y_test[target].tolist()


    # print('Treinando modelo...\n')
    # d2v_model = model.train_doc2vec_embeddings(vec_size, model_option, X_train)

    print('Carregando modelo...\n')
    d2v_model = model.get_model('saved_models/doc2vec.model')

    X_train = model.infer_d2v_embeddings(d2v_model, X_train)
    X_test = model.infer_d2v_embeddings(d2v_model, X_test)


    arq = 'Regress_90_10_'+target+'.txt'
    f = open(arq, 'a')

    
    regrs = []
    regrs.append((linear_model.LinearRegression(), 'LinearRegression'))
    regrs.append((linear_model.SGDRegressor(max_iter=10000), 'SGD Regression'))
    regrs.append((linear_model.HuberRegressor(max_iter=10000), 'HurberRegression (Regressão robusta)'))
    regrs.append((linear_model.Lasso(), 'Lasso'))
    regrs.append((linear_model.ElasticNet(), 'Elastic-Net'))
    regrs.append((linear_model.Ridge(), 'Ridge'))
    regrs.append((linear_model.BayesianRidge(), 'Regressão Bayesiana'))
    regrs.append((KernelRidge(), 'Kernel ridge regression'))
    regrs.append((KNeighborsRegressor(n_neighbors=3), 'k-nearest neighbors'))
    regrs.append((GaussianProcessRegressor(), 'Regressão gaussiana'))
    regrs.append((tree.DecisionTreeRegressor(), 'Árvore de regressão'))
    regrs.append((RandomForestRegressor(), 'Randon forest'))
    regrs.append((ExtraTreesRegressor(), 'Extra Trees Regressor'))
    regrs.append((GradientBoostingRegressor(), 'Gradiente Boost'))
    regrs.append((MLPRegressor(activation='relu',max_iter=100000, n_iter_no_change= 50,solver='adam', tol=0.001), 'MLP'))

    print('Treinando regressores...\n')

    for regr in tqdm(regrs):
        f.write('\n\n--------------> '+regr[1]+' <-----------------------\n')  
        
        QWK, MSE, classif_report = model.run_model(regr[0], X_train, X_test, y_train, y_test)
        
        f.write('\nKappa: {0:.4f}'.format(QWK))
        f.write('\nMSE: {0:.4f}'.format(MSE))
        
        f.write('\n----------------------------------------------------------\n')

    f.close()



