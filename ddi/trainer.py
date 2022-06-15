from google.cloud import storage
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
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
    forest = RandomForestClassifier(n_estimators=40, random_state=1, criterion= 'gini' )
    clf = MultiOutputClassifier(forest, n_jobs = -1)
    clf.fit(X_train,y_train)
    return clf

def test(X_test,y_test):
    '''evaluate the model's performance on test data'''
    y_pred = clf.predict(X_test)
    accuracy = 1 - hamming_loss(y_test,y_pred)
    print(f"accuracy: {accuracy}")
    return accuracy

def train_full(df):
    '''train the model on the full dataset'''
    X = df[df.columns[89:]]
    y = df[df.columns[3:89]]

    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    pca = PCA(n_components = 150)
    pca.fit(X)
    X = pca.transform(X)
    model = train(X,y)

    return pca, model

def save_model_joblib(model):
    '''saving models'''
    joblib.dump(model, 'model.joblib', compress = 3)
    print("saved model.joblib locally")

def save_model_pca(model):
    joblib.dump(model,'pca.joblib')
    print("saved pca.joblib locally")

if __name__ == "__main__":
    df = get_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    clf = train(X_train,y_train)
    test(X_test,y_test)
    pca, model = train_full(df)
    save_model_joblib(model)
    save_model_pca(pca)
    size_bytes = os.stat('../model.joblib',).st_size
    print(f"size_bytes is {size_bytes}")
