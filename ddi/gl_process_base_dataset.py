import pandas as pd
import numpy as np
from ddi.utils import get_data_filepath, get_rawdata_filepath
from ddi.encoders import reclassification_encoder
import warnings
warnings.filterwarnings('ignore')


def df_optimized(df, verbose=False):
    '''Reduce memory usage of a dataframe'''
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            df[col] = round(df[col],4)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    df.replace({False: 0, True: 1}, inplace=True) # converting bool into int
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


def drug_map(ts_df):
    '''Returns a dictionary with keys as the SMILES strings and values ranging from 100 to
    745 as a unique identifier number
    '''
    total_smiles = ts_df['Drug1'].append(ts_df['Drug2']) # combining both drug1 and drug2 smiles together
    unique_smiles = (set(list(total_smiles))) #create a list of drug_names and extract unique smiles
    unique_smiles = pd.DataFrame(unique_smiles).rename(columns ={0:'Drug'})# create a dataframe for label encoding purposes
    unique_smiles.sort_values(by='Drug', inplace = True)
    unique_smiles['Drug_ID'] = [i for i in range(100,745)] # manual encode them starting from 100 to 745
    drug_dict = pd.Series(unique_smiles.Drug_ID.values, index = unique_smiles.Drug).to_dict()# forming a dictionary to map
    return drug_dict


def clean_twosides(ts_df):
    '''Perform memory optimization and cleaning up of ts_df'''
    # mapping drug numbers to smiles
    ts_df['Drug1'] = ts_df['Drug1'].map(drug_dict)
    ts_df['Drug2'] = ts_df['Drug2'].map(drug_dict)

    # drop irrelavant columns
    ts_df = ts_df.drop(columns = ['Unnamed: 0', 'Drug1_ID','Drug2_ID'])

    # creating unique drug ID for drug 1 and drug 2 combination
    DD_ID =  ts_df['Drug1'].astype(str) + ts_df['Drug2'].astype(str)
    DD_ID = DD_ID.astype('int32')
    ts_df.insert(loc=2,column ='DD_ID', value = DD_ID)

    # optimized the df
    df_optimized(ts_df)

    return ts_df.sort_values(by =['DD_ID'])


def ts_pivot(ts_df):
    '''Reshaping ts_df to generate 35 additional columns for each of the side effects'''
    ts_df = ts_df.merge(reclass, on='Y', how='left')  # merging reclassification
    pivot = ts_df.pivot(columns ='Y_cat',values ='DD_ID').fillna(0).astype('int32') ## pivot Y_cat into columns
    pivot_df = pd.concat([ts_df,pivot], axis = 1) # concat ts_df to and pivot df
    pivot_df = pivot_df.drop(columns = ['Drug1','Drug2','Y','Y_cat']).groupby('DD_ID').sum() ## dropping irrelavant columns and groupby by unique ID

    # transforming dataframe into binary
    for series in pivot_df.columns:
        pivot_df[series] = pivot_df[series].apply(lambda x: x if x == 0 else 1)

    # slicing the index = DD_ID, into drug 1 and drug 2
    drug1 = pivot_df.index // 1000
    drug2 = pivot_df.index % 1000

    # Adding back drug 1 and 2 into df
    pivot_df.insert(loc=0, column='Drug1', value=drug1)
    pivot_df.insert(loc=1, column='Drug2', value=drug2)

    return pivot_df.reset_index()


if __name__ == '__main__':
    print('Creating base dataset for project phase 2...')

    # Importing Files
    ts_df = pd.read_csv(get_rawdata_filepath('twosides.csv'))
    try:
        reclass = pd.read_csv(get_data_filepath('gl_reclassification_encoded.csv'))
    except:
        reclassification_encoder('gl_reclassification_encoded.csv')
        reclass = pd.read_csv(get_data_filepath('gl_reclassification_encoded.csv'))
    try:
        feat_eng_df = pd.read_csv(get_data_filepath('cxe_feat_eng.csv'))
    except:
        reclassification_encoder('cxe_feat_eng.csv')
        feat_eng_df = pd.read_csv(get_data_filepath('cxe_feat_eng.csv'))

    # Simple Cleaning
    reclass = reclass[['Y','Y_cat']]

    # Running above functions
    drug_dict = drug_map(ts_df)
    ts_df = clean_twosides(ts_df)
    pivot_df = ts_pivot(ts_df)

    # ## Merging with drug feature
    # mapping smiles into numbers using drug_dict
    feat_eng_df['smiles'] = feat_eng_df['smiles'].map(drug_dict)
    # renaming column for better clarity
    feat_eng_df.rename(columns = {'smiles': 'Drug'}, inplace=True)
    # merging with the main dataframe with drug1 features
    df = pivot_df.merge(feat_eng_df, left_on='Drug1', right_on='Drug', how = 'left')

    # transforming feat_eng_df features by adding _1 at the ... cont
    # back of each feature so that we can differentiate with drug1 features,
    # after merging the main dataframe with drug2 features
    for col in feat_eng_df.columns[1:]:
        feat_eng_df.rename(columns = {col: col+'_1'}, inplace=True)
    df = df.merge(feat_eng_df, left_on='Drug2', right_on='Drug', how = 'left')

    # dropping additional columns generated from the merging step
    df.drop(columns=['Drug_x', 'Drug_y'], inplace=True)

    # ## Differencing the features
    f_list = list(df.columns[38:759]) # list of features to perform differencing iteration
    for col in f_list: # iterating over each feature name
        df[col+'_diff']= df[col].sub(df[col+'_1']) # creating a new column for differenced features
        df.drop(columns = [col,col+'_1'], inplace = True) # dropping two columns

    final_df = df_optimized(df)

    # ## Exporting final_df
    final_df.to_csv(get_data_filepath('gl_base_dataset.csv'))
    print("gl_base_dataset.csv created and stored in data folder")
