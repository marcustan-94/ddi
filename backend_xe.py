from mordred import Calculator, descriptors
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
import joblib


Y_class= pd.read_csv("raw_data/mt_reclassification_encoded_beautify.csv",
                   usecols = ['sub_system_severity','Y_cat'],
                    nrows = 1307) # remove rows

df = pd.read_csv('raw_data/base_diff_df.csv', nrows=0)
df.drop(columns =[col for col in df.columns if 'Unnamed' in col], inplace = True )
X = df[df.columns[90:]]


def get_smiles(drug1,drug2):

    '''converting drug names to smiles, return try again error if cant process the drug names'''
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
    smile_list = get_smiles(drug1, drug2)

    '''building pandas dataframe'''
    mols = [Chem.MolFromSmiles(item) for item in smile_list]
    calc = Calculator(descriptors, ignore_3D=True)
    drug_features = calc.pandas(mols)
    return drug_features


def preproc(drug1, drug2):
    '''cleaning drug_features dataframe'''
    drug_features = get_mordred(drug1, drug2)
    drug_features.replace({False: 0, True: 1}, inplace=True)
    drug_features = drug_features[X.columns]
    drug_features.iloc[0] = drug_features.iloc[0].astype("float32")
    drug_features.iloc[1] = drug_features.iloc[1].astype("float32")
    X_test = pd.DataFrame(drug_features.iloc[0] - drug_features.iloc[1]).astype('float32').transpose()
    return X_test


def load_model():
    '''predicting X_test with the model'''
    pipeline = joblib.load('model.joblib')
    return pipeline

def predict(drug1, drug2):
    pipeline = load_model()
    X_test = preproc(drug1, drug2)
    y_pred = pipeline.predict(X_test)
    return y_pred

def predict_proba(drug1, drug2):
    pipeline = load_model()
    X_test = preproc(drug1, drug2)
    y_proba = pipeline.predict_proba(X_test)
    return y_proba



def classify(drug1, drug2):
    y_pred = predict(drug1, drug2)
    '''creating a dictionary that returns the sub_system as values '''
    cat_dict = pd.Series(Y_class.sub_system_severity.values, index = Y_class.Y_cat).to_dict()

    '''retrieving the predicted categories and store into a list'''
    prediction_list = []
    for i,x in enumerate(y_pred[0]):
        if x == 0:
            continue
        prediction_list.append(i)

    '''retrieving the side_effects given the predicted categories'''
    side_effect_list = []
    for i in prediction_list:
        side_effect_list.append(cat_dict[i])

    return side_effect_list

def classify_proba(drug1, drug2):
    y_pred = predict(drug1, drug2)
    y_proba = predict_proba(drug1, drug2)
    prediction_list = []
    proba_list = []
    for i,x in enumerate(y_pred[0]):
        if x == 0:
            continue
        prediction_list.append(i)

    for i in prediction_list:
        proba_list.append(y_proba[i][0][1])

    return proba_list
