from google.cloud import storage
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
import warnings
warnings.filterwarnings('ignore')

from ddi.utils import df_optimized, get_data_filepath

def get_data():
    '''retrieve and clean the final_dataset'''
    df = pd.read_csv(get_data_filepath('final_dataset.csv'))
    df.drop(columns =[col for col in df.columns if 'Unnamed' in col], inplace = True)
    df.drop(columns = ['26','87'], inplace = True)
    df = df_optimized(df)
    return df

def preprocess(df):
    '''transform the df into pca features'''
    X = df[df.columns[89:]]  # check with marcus what are X and y columns -
    y = df[df.columns[3:89]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3 )

    '''scaling_train X'''
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train = pd.DataFrame(std_scaler.transform(X_train), columns = X.columns)

    '''pca_transform'''
    pca = PCA(n_components = 150)
    pca.fit(X_train)
    X_train = pca.transform(X_train)

    X_test = std_scaler.transform(X_test)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test

def train(X_train,y_train):
    '''train the model'''
    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    clf = MultiOutputClassifier(forest, n_jobs = -1)
    clf.fit(X_train,y_train)
    return clf

def test(X_test,y_test):
    '''evaluate the model's performance on test data'''
    y_pred = clf.predict(X_test)
    accuracy = 1 - hamming_loss(y_test,y_pred)
    print('accuracy')
    return accuracy

def train_full(df):
    '''train the model on the full dataset'''
    X = df[df.columns[89:]]  # check with marcus what are X and y columns -
    y = df[df.columns[3:89]]

    std_scaler = StandardScaler()
    std_scaler.fit_transform(X)
    pca = PCA(n_components = 150)
    pca.fit(X)
    X = pca.transform(X)
    model = train(X,y)

    return pca, model

    # forest = RandomForestClassifier(n_estimators=100, random_state=1)
    # clf = MultiOutputClassifier(forest, n_jobs = -1)
    # clf.fit(X,y)
    # return clf

def save_model(model):
    # saving the trained model to disk is mandatory to be able to upload it to storage
    if model == 'model':
        joblib.dump(model, 'model.joblib')
        print("saved model.joblib locally")
    else:
        joblib.dump(model,'pca.joblib')
        print("saved pca.joblib locally")

if __name__ == "__main__":
    df = get_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    clf = train(X_train,y_train)
    test(X_test,y_test)
    pca, model = train_full(df)
    save_model(model)
    save_model(pca)
