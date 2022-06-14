import pandas as pd
import numpy as np
from ddi.utils import get_data_filepath, get_rawdata_filepath, df_optimized
import warnings
warnings.filterwarnings('ignore')


def drug_map(ts_df):
    '''
    Returns a dictionary with keys as the SMILES strings and values ranging from 100 to
    745 as a unique identifier number
    '''
    # combining both drug 1 and drug 2 smiles together and creating a list of unique smiles
    total_smiles = ts_df['Drug1'].append(ts_df['Drug2'])
    unique_smiles = (set(list(total_smiles)))
    # converting the unique smiles list to a dataframe for label encoding purposes
    unique_smiles = pd.DataFrame(unique_smiles).rename(columns = {0:'Drug'})
    unique_smiles.sort_values(by='Drug', inplace=True)
    # assigning each drug smiles a unique identifier number from 100 to 745
    unique_smiles['Drug_ID'] = [i for i in range(100,745)]
    # converting the dataframe back into a dictionary
    drug_dict = pd.Series(unique_smiles['Drug_ID'].values, index = unique_smiles['Drug']).to_dict()

    return drug_dict


def clean_twosides(ts_df):
    '''
    Replacing drug smiles with their respective unique identifier numbers.
    Perform memory optimization and cleaning up ts_df
    '''
    # replacing smiles strings with the unique drug numbers as defined in the drug_dict
    ts_df['Drug1'] = ts_df['Drug1'].map(drug_dict)
    ts_df['Drug2'] = ts_df['Drug2'].map(drug_dict)
    # drop irrelavant columns
    ts_df = ts_df.drop(columns = ['Drug1_ID','Drug2_ID'])
    # creating unique 6-digit DD_ID for each drug 1 and drug 2 combination
    DD_ID = ts_df['Drug1'].astype(str) + ts_df['Drug2'].astype(str)
    DD_ID = DD_ID.astype('int32')
    ts_df.insert(loc=2, column='DD_ID', value=DD_ID)
    # performing memory optimization
    df_optimized(ts_df)

    return ts_df.sort_values(by =['DD_ID'])


def ts_pivot(ts_df):
    '''Reshaping ts_df to generate columns for each of the side effect classes'''
    # merging Y_cat column from reclass_df with ts_df
    ts_df = ts_df.merge(reclass_df, on='Y', how='left')
    # pivot Y_cat into columns
    pivot = ts_df.pivot(columns='Y_cat',values='DD_ID').fillna(0).astype('int32')
    # concat ts_df with pivoted Y_cat columns
    pivot_df = pd.concat([ts_df, pivot], axis=1)
    # dropping irrelavant columns and groupby by DD_ID
    pivot_df = pivot_df.drop(columns=['Drug1','Drug2','Y','Y_cat']).groupby('DD_ID').sum()

    # transforming each side effect column into binary
    for series in pivot_df.columns:
        pivot_df[series] = pivot_df[series].apply(lambda x: x if x == 0 else 1)

    # Re-obtaining the unique identifier numbers of drug 1 and 2, by slicing the DD_ID
    drug1 = pivot_df.index // 1000
    drug2 = pivot_df.index % 1000

    # Adding back drug 1 and 2 unique identifier numbers into pivot_df
    pivot_df.insert(loc=0, column='Drug1', value=drug1)
    pivot_df.insert(loc=1, column='Drug2', value=drug2)

    return pivot_df.reset_index()


if __name__ == '__main__':
    print('Creating final_dataset.csv...')

    # importing files
    ts_df = pd.read_csv(get_rawdata_filepath('twosides.csv'))
    reclass_df = pd.read_csv(get_data_filepath('complete_severity_reclassification.csv'))
    feat_eng_df = pd.read_csv(get_data_filepath('feature_engineering.csv'))

    # simple cleaning
    reclass_df = reclass_df[['Y','Y_cat']]

    # running above functions
    drug_dict = drug_map(ts_df)
    ts_df = clean_twosides(ts_df)
    pivot_df = ts_pivot(ts_df)


    # ## Merging with feat_eng_df
    # converting each drug smiles into their respective unique drug identifier numbers, using drug_dict
    feat_eng_df['smiles'] = feat_eng_df['smiles'].map(drug_dict)
    # renaming column for better clarity
    feat_eng_df.rename(columns = {'smiles': 'Drug'}, inplace=True)
    # storing feat_eng_df column names in feat_eng_df_cols variable as it will be needed later
    feat_eng_df_cols = feat_eng_df.columns[1:]

    # merging pivot_df with drug1 and drug2 features
    # feature names for drug2 will contain _1 at the back to differentiate it from drug1 features
    final_df = pivot_df.merge(feat_eng_df, left_on='Drug1', right_on='Drug', how = 'left')
    for col in feat_eng_df.columns[1:]:
        feat_eng_df.rename(columns = {col: col+'_1'}, inplace=True)
    final_df = final_df.merge(feat_eng_df, left_on='Drug2', right_on='Drug', how = 'left')

    # dropping additional columns generated from the merging step
    final_df.drop(columns=['Drug_x', 'Drug_y'], inplace=True)

    # ## Differencing the features
    for col in feat_eng_df_cols:
        final_df[col+'_diff']= final_df[col].sub(final_df[col+'_1']) # creating a new column for differenced features
        final_df.drop(columns = [col,col+'_1'], inplace = True) # dropping unneeded columns

    # obtaining absolute values as we are only interested in the magnitude of the difference between features
    final_df = final_df.abs()

    # performing memory optimization
    final_df = df_optimized(final_df)

    # ## Exporting final_df
    final_df.to_csv(get_data_filepath('final_dataset.csv'))
    print("final_dataset.csv created and stored in data folder")
