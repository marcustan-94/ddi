import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from math import floor, ceil

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ddi.utils import df_optimized, get_data_filepath


hamming_loss_neg = make_scorer(lambda y_true, y_pred: 1-hamming_loss(y_true,y_pred))

def get_data():
    '''retrieve and clean the final_dataset'''
    df = pd.read_csv(get_data_filepath('final_dataset.csv'))
    df.drop(columns =[col for col in df.columns if 'Unnamed' in col], inplace = True)
    df.drop(columns = ['86'], inplace = True)
    df = df_optimized(df)
    return df


def pre_process(df):
    X = abs(df[df.columns[89:]])
    y = df[df.columns[3:89]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    '''scaling X_train'''
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train = pd.DataFrame(std_scaler.transform(X_train), columns = X.columns)

    '''Find the number of principal components that explains 80% of the variation in the dataset'''
    pca = PCA()
    pca.fit(X_train)

    pc = 0
    for index, var in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        if var > 0.80:
            pc = index
            print(f" 80% of the variation can be explained by {index} principal components")
            break

    '''transform both X_train and Y_train'''
    pca = PCA(n_components = pc)
    pca.fit(X_train)
    x_train = pca.transform(X_train)

    X_test = std_scaler.transform(X_test)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test


def random_grid_search(X_train,y_train):
    """input X and Y train to return best parameters through a random grid search"""
    rfc = RandomForestClassifier()
    clf = MultiOutputClassifier(rfc)

    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
    max_depth = [int(x) for x in np.linspace(10, 20, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]


    random_grid = {'estimator__n_estimators': n_estimators,
                   'estimator__max_depth': max_depth,
                   'estimator__min_samples_split': min_samples_split,
                   'estimator__min_samples_leaf': min_samples_leaf
                  }

    search = RandomizedSearchCV(
        estimator = clf,
        n_iter = 50,
        param_distributions = random_grid,
        cv = 3,
        n_jobs = -3,
        verbose = 1,
        scoring = hamming_loss_neg )

    search.fit(X_train, y_train)

    return search.best_params_

def grid_search(X_train,y_train):


    rfc = RandomForestClassifier()
    clf = MultiOutputClassifier(rfc)

    """return the random search best parameters """
    random_search_params = random_grid_search(X_train,y_train)

    ''' create a range of values for each hyperparameter based on random search best parameters'''
    param_grid = {}
    for k,v in random_search_params.items():
        param_grid[k] = [floor(v*0.9), v, ceil(v*1.1)]


    search = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        cv = 3,
        n_jobs = -2,
        verbose = 1,
        scoring = hamming_loss_neg)

    search.fit(X_train, y_train)

    return search.best_params_

if __name__ == "__main__":
    df = get_data()
    X_train, X_test, y_train, y_test = pre_process(df)
    best_params = grid_search(X_train,y_train)
    print(f"best estimators are {best_params}")
