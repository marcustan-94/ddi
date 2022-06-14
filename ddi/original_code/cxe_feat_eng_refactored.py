import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from ddi.utils import get_data_filepath, get_rawdata_filepath
import warnings
warnings.filterwarnings('ignore')

class RawData:
    def __init__(self, file_name):
        self.file_name = file_name

    def get_data(self):
        '''opening twosides file as df'''
        twosides_df = pd.read_csv(self.file_name, index_col=0)
        return twosides_df

    def get_smiles_list(self):
        '''getting a list of unique smiles strings'''
        twosides_df = self.get_data()
        smiles_list = list(set(twosides_df['Drug1'].unique().tolist() + twosides_df['Drug2'].unique().tolist()))
        return smiles_list

    def build_features(self):
        '''Returns a dataframe containing engineered features for each SMILES string. '''

        # initiating mordred calculator
        calc = Calculator(descriptors, ignore_3D=True)
        # getting drug SMILES list
        smiles_list = self.get_smiles_list()
        # converting SMILES string into a molecular form for fitting into the mordred calculator
        mols = [Chem.MolFromSmiles(item) for item in smiles_list]
        # building dataframe of engineered features
        drug_features = calc.pandas(mols)
        # drug_features dataframe lacks the SMILES string so adding in t
        drug_features['smiles'] = smiles_list
        # shifting SMILES column to the front so it is easier to visualize
        drug_features = drug_features.set_index('smiles').reset_index()

        # building the list of columns to drop by selecting dtypes = "object",
        # as we only want the float/int dtypes

        # the list doesn't include the SMILES column as we need the SMILE structures for merging'''
        drug_features_object_list = list(drug_features.select_dtypes('object').keys()[1:])
        # dropping the columns where dtypes = "objects", except the SMILES column
        drug_features = drug_features.drop(columns=drug_features_object_list)

        # replace Booelan series to binary
        drug_features.replace({False: 0, True: 1}, inplace=True)

        return drug_features

    def convert_to_csv(self, csv_name):
        '''converting the file to csv'''
        drug_features_df = self.build_features()
        drug_features_df.to_csv(csv_name, index=False)
        print("cxe_feat_eng.csv created and stored in data folder")

if __name__ == '__main__':
    print('Engineering features from chemical structures...')
    file_name = get_rawdata_filepath('twosides.csv')
    raw_data = RawData(file_name)
    raw_data.convert_to_csv(get_data_filepath('cxe_feat_eng.csv'))
