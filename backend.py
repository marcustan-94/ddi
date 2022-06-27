import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote

from ddi.utils import get_data_filepath


# The target classes (sub system severity) are loaded as dataset to allow reclassification of predictions
# from numbers into words. As the class number 86 is miscellaneous, the class is dropped.
Y_class = pd.read_csv(get_data_filepath('complete_severity_reclassification.csv'),
                      usecols = ['sub_system_severity','Y_cat'])
Y_class = Y_class[(Y_class['Y_cat'] != 86)]

# Mordred calculates over 1300 molecular features of the drugs, but as only 721
# features are taken into consideration, these feature column names are loaded
# so that only these features are taken into consideration from all drug features
feat_eng_df = pd.read_csv(get_data_filepath('feature_engineering.csv'), nrows=0)
X = feat_eng_df[feat_eng_df.columns[1:]]


def get_smiles(drug1,drug2):
    '''Converts drug names to smiles structures, returns try again
    error if the drug names cannot be converted by the API
    '''
    smile_list = []
    for drug in [drug1,drug2]:
        try:
            url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(drug) + '/smiles'
            ans = urlopen(url).read().decode('utf8')
            smile_list.append(ans)
        except:
            smile_list.append('Unable to find the drug, please try again.')

    return smile_list


def get_mordred(drug1, drug2):
    '''Calculates the molecular features of the drugs from their
    smile structures and builds them into a dataframe
    '''
    # Drug names are converted into smiles through API
    smile_list = get_smiles(drug1, drug2)

    # Molecular features dataframe is built from the smile list provided
    mols = [Chem.MolFromSmiles(item) for item in smile_list]
    calc = Calculator(descriptors, ignore_3D=True)
    drug_features = calc.pandas(mols)
    return drug_features


def preproc(drug1, drug2):
    '''Preprocesses and cleans the drug_features dataframe'''
    drug_features = get_mordred(drug1, drug2)
    # Booleans in the dataframe are converted into numbers 0 and 1
    drug_features.replace({False: 0, True: 1}, inplace=True)
    # Only the feature columns which are taken into consideration will be used
    drug_features = drug_features[X.columns]
    # All features in the dataframe are now numbers, but some are represented by
    # dtype string. These number dtypes are converted into float
    drug_features.iloc[0] = drug_features.iloc[0].astype("float32")
    drug_features.iloc[1] = drug_features.iloc[1].astype("float32")
    # Features are differenced to calculate the similarity of molecular features between the drug pair
    X_test = pd.DataFrame(drug_features.iloc[0] - drug_features.iloc[1]).astype('float32').transpose()
    # obtaining absolute values as we are only interested in the magnitude of the difference between features
    X_test = X_test.abs()
    # PCA is performed to reduce the dimensionality of the differenced features
    pca = joblib.load('pca.joblib')
    X_test = pca.transform(X_test)

    return X_test


def predict(drug1, drug2, model):
    '''Predicts the target class numbers from the input drug combination'''
    # Features dataframe is built
    X_test = preproc(drug1, drug2)
    # Model is used to predict the target class numbers
    y_pred = model.predict(X_test)
    return y_pred


def classify(drug1, drug2, model):
    '''Converts the predicted class numbers into names of classes predicted'''
    y_pred = list(predict(drug1, drug2, model)[0])
    # A dictionary is created that returns the predicted sub_system as values
    y_dict = pd.Series(Y_class.sub_system_severity.values, index = Y_class.Y_cat).to_dict()

    #The predicted classes are retrieved and stored into a list'''
    prediction_list = []
    for i, x in enumerate(y_pred):
        if x == 0:
            continue
        prediction_list.append(i)

    #The predicted side effects are retrieved given the predicted categories'''
    side_effect_list = []
    for i in prediction_list:
        side_effect_list.append(y_dict[i])

    return side_effect_list

if __name__ == "__main__":
    model = load_model()
    print(classify('Aspirin','Paracetamol', model))
