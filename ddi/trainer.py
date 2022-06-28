import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ddi.utils import df_optimized, get_data_filepath


def get_data():
    '''Retrieve, clean and optimize memory usage of the final_dataset'''
    df = pd.read_csv(get_data_filepath('final_dataset.csv'))
    df.drop(columns =[col for col in df.columns if 'Unnamed' in col], inplace = True)
    df.drop(columns = ['86'], inplace = True)
    df = df_optimized(df)
    return df


def preprocess(df):
    '''transform the df into pca features'''
    X = df[df.columns[89:]]  # check with marcus what are X and y columns -
    y = df[df.columns[3:89]]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3 )


    # creatin a pipeline to process the data
    # 80% of the variation can be explained by 46 principal components from param_search.py
    preproc = make_pipeline(StandardScaler(),PCA(n_components =46))
    preproc.fit(X_train)
    X_train = preproc.transform(X_train)
    X_test = preproc.transform(X_test)

    return preproc, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    '''Train the model on train dataset'''
    forest = RandomForestClassifier(n_estimators=10, random_state=1, criterion='gini')
    clf = MultiOutputClassifier(forest, n_jobs =-3)
    clf.fit(X_train, y_train)
    return clf


def test(X_test, y_test):
    '''Evaluate the model's performance on test dataset, using hamming loss as the accuracy metric'''
    y_pred = clf.predict(X_test)
    accuracy = 1 - hamming_loss(y_test, y_pred)
    print(f"accuracy: {accuracy}")


def train_full(df):
    '''Train the model on the full dataset'''
    X = df[df.columns[89:]]
    y = df[df.columns[3:89]]

    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    pca = PCA(n_components = 46)
    X = pca.fit_transform(X)
    model = train(X, y)

    return pca, model


def save_model_joblib(model):
    '''Saving Random Forest Classifier model'''
    joblib.dump(model, 'model.joblib', compress=5)
    print("saved model.joblib locally")


def save_preproc(model):
    '''saving prepoc model'''
    joblib.dump(model,'preproc.joblib')
    print("saved preproc.joblib locally")


if __name__ == "__main__":
    df = get_data()
    preproc, X_train, X_test, y_train, y_test = preprocess(df)
    clf = train(X_train,y_train)
    test(X_test,y_test)
    save_model_joblib(clf)
    save_preproc(preproc)
    size_bytes = os.stat('model.joblib',).st_size
    print(f"size_bytes is {size_bytes}.")
