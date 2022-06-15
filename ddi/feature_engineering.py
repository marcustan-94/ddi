import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from ddi.utils import get_data_filepath, get_rawdata_filepath
import warnings
warnings.filterwarnings('ignore')


def build_features(ts_df, unique_smiles):
    '''Returns a dataframe containing engineered features for each smiles string.'''

    # Using mordred to calculate physicochemical properties from the smiles string and store the data in a df
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]
    feat_eng_df = calc.pandas(mols)

    # Dropping columns of 'object' dtype, as we only want the float/int dtypes
    object_list = list(feat_eng_df.select_dtypes('object').keys())
    feat_eng_df = feat_eng_df.drop(columns=object_list)

    # replacing Booelan series to binary
    feat_eng_df.replace({False: 0, True: 1}, inplace=True)

    # adding 'smiles' column to feat_eng_df
    feat_eng_df.insert(loc=0, column='smiles', value=unique_smiles)

    return feat_eng_df


if __name__ == '__main__':
    print('Engineering features from chemical structures...')

    # import ts_df
    ts_df = pd.read_csv(get_rawdata_filepath('twosides.csv'))
    # getting a list of unique smiles strings
    unique_smiles = list(set(ts_df['Drug1'].unique().tolist() + ts_df['Drug2'].unique().tolist()))
    # creating feat_eng_df containing engineered features for each smiles string
    feat_eng_df = build_features(ts_df, unique_smiles)
    # exporting to csv
    feat_eng_df.to_csv(get_data_filepath('feature_engineering.csv'), index=False)

    print("feature_engineering.csv created and stored in data folder")
