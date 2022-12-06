
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree, linear_model  
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, MaxAbsScaler

import pickle

from tqdm import tqdm

import pandas as pd


def test_model(f, X_train, y_train, X_test, y_test, regr, trat = None):
    if trat == 'norm':
        # Normalização
        X_train = normalize(X_train, norm='l2', axis=0)
        X_test = normalize(X_test, norm='l2', axis=0)
    elif trat == 'padr':
        # Padroniação z-score
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    elif trat == 'trans':
        # transformação MinMax
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
    elif trat == 'trans_gaus':
        # Transformação para uma distribuição gaussiana
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X_train = pt.fit_transform(X_train)
        X_test = pt.fit_transform(X_test)
    elif trat == 'trans_uni':
        # Transformação para uma distuibuição uniforme
        quantile_transformer = QuantileTransformer(random_state=0)
        X_train = quantile_transformer.fit_transform(X_train)
        X_test = quantile_transformer.transform(X_test)
    elif trat == 'trans_autl':
        # Transformação para uma distuibuição com outliers
        quantile_transformer = RobustScaler(with_centering=True)
        X_train = quantile_transformer.fit_transform(X_train)
        X_test = quantile_transformer.transform(X_test)
    elif trat == 'maxabs':
        # Transformação para escalar dados esparsos
        transformer = MaxAbsScaler().fit(X_train)
        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)


    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test) 

    y_cat = []
    for t in y_pred:
        if t < 60:
            y_cat.append(40)
        elif 60 <= t < 100:
            y_cat.append(80)
        elif 100 <= t < 140:
            y_cat.append(120)
        elif 140 <= t < 180:
            y_cat.append(160)
        elif 180 <= t:
            y_cat.append(200)
    y_pred = y_cat
    
    QWK = cohen_kappa_score(y_test, y_pred, weights = 'quadratic')
    
    f.write('\nKappa: {0:.4f}'.format(QWK))
    
    
if __name__ == '__main__':
    with open('X_train.pickle', 'rb') as file_pickle:
        X_train = pickle.loads(file_pickle.read())
        file_pickle.close()
    with open('y_train.pickle', 'rb') as file_pickle:
        y_train = pickle.loads(file_pickle.read())
        file_pickle.close()
    with open('X_test.pickle', 'rb') as file_pickle:
        X_test = pickle.loads(file_pickle.read())
        file_pickle.close()
    with open('y_test.pickle', 'rb') as file_pickle:
        y_test = pickle.loads(file_pickle.read())
        file_pickle.close()


    f = open('resultados.txt', 'a')

    
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


    for regr in tqdm(regrs):
        f.write('\n\n--------------> '+regr[1]+' <-----------------------\n')  
        
        test_model(f, X_train, y_train, X_test, y_test, regr[0])
        
        f.write('\n\n>> Normalização\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='norm')
        

        f.write('\n\n>> Padroniação z-score\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='padr')


        f.write('\n\n>> Transformação MinMax\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='trans')


        f.write('\n\n>> Transformação para uma distribuição gaussiana\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='trans_gaus')


        f.write('\n\n>> Transformação para uma distuibuição uniforme\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='trans_uni')


        f.write('\n\n>> Transformação para uma distuibuição com outliers\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='trans_autl')

        f.write('\n\n>> Transformação para escalar dados esparsos\n')
        test_model(f, X_train, y_train, X_test, y_test, regr[0], trat='maxabs')

        f.write('\n----------------------------------------------------------\n')

    f.close()



