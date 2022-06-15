
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ## Functions
def df_optimized(df, verbose=True):
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


# ## Importing Files
ts_df = pd.read_csv("/Users/george/Desktop/LW-DDI Project/twosides.csv")
reclass = pd.read_csv("/Users/george/Desktop/LW-DDI Project/gl_reclassification_encoded.csv")
fe_drug1 = pd.read_csv("/Users/george/Desktop/LW-DDI Project/cxe_feat_eng_drug1_droppedcolumns.csv")
fe_drug2 = pd.read_csv("/Users/george/Desktop/LW-DDI Project/cxe_feat_eng_drug2_droppedcolumns.csv")


# ## Simple Cleaning
reclass = reclass[['Y','Y_cat']]
fe_drug1 = fe_drug1.drop(columns = 'Unnamed: 0')
fe_drug2 = fe_drug2.drop(columns ='Unnamed: 0')


# ## Encoding
def drug_map(ts_df):
    total_smiles = ts_df['Drug1'].append(ts_df['Drug2']) # combining both drug1 and drug2 smiles together
    unique_smiles = (set(list(total_smiles))) #create a list of drug_names and extract unique smiles
    unique_smiles = pd.DataFrame(unique_smiles).rename(columns ={0:'Drug'})# create a dataframe for label encoding purposes
    unique_smiles.sort_values(by='Drug', inplace = True)
    unique_smiles['Drug_ID'] = [i for i in range(100,745)] # manual encode them starting from 100 to 745
    drug_dict = pd.Series(unique_smiles.Drug_ID.values, index = unique_smiles.Drug).to_dict()# forming a dictionary to map
    return drug_dict

drug_dict =drug_map(ts_df)

# ## Clean twosides
def clean_twosides(ts_df):
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
ts_df = clean_twosides(ts_df)


# ## Reshaping
def ts_pivot(ts_df):
    ts_df = ts_df.merge(reclass, on='Y', how='left')  # merging reclassification
    pivot = ts_df.pivot(columns ='Y_cat',values ='DD_ID').fillna(0).astype('int32') ## pivot Y_cat into columns
    pivot_df = pd.concat([ts_df,pivot], axis = 1) # concat ts_df to and pivot df
    pivot_df = pivot_df.drop(columns = ['Drug1','Drug2','Y','Y_cat']).groupby('DD_ID').sum() ## dropping irrelavant columns and groupby by unique ID

    # transforming dataframe into binary
    for series in pivot_df.columns:
        pivot_df[series] = pivot_df[series].apply(lambda x: x if x == 0 else 1)

    # slicing the index = DD_ID, into drug 1 and drug 2
    Drug1 = pivot_df.index // 1000
    Drug2 = pivot_df.index % 1000

    # Adding back drug 1 and 2 into df
    pivot_df.insert(loc=0,column ='Drug1', value = Drug1)
    pivot_df.insert(loc=1, column='Drug2',value=Drug2)

    return pivot_df.reset_index()

pivot_df = ts_pivot(ts_df)


# ## Merging with drug feature
# mapping smiles into numbers using drug_dict
fe_drug1['Drug1'] = fe_drug1['Drug1'].map(drug_dict)
fe_drug2['Drug2'] = fe_drug2['Drug2'].map(drug_dict)

# transforming fe_drug2 features by adding _1 at the ... cont
# back of each feature so that we can differentiate with drug1 features.
for col in fe_drug2.columns[1:]:
    fe_drug2.rename(columns = {col: col+'_1'}, inplace=True)

# merging with the main dataframe with the drug1 and drug2 features
df = pivot_df.merge(fe_drug1, on ='Drug1', how = 'left')
df = df.merge(fe_drug2, on ='Drug2', how = 'left')

# replace Booelan series to binary
df.replace({False: 0, True: 1}, inplace=True)

# differencing the features
f_list = list(fe_drug1.columns[1:]) # list of features to perform differencing iteration
for col in f_list: # iterating over each feature name
    df[col+'_diff']= df[col].sub(df[col+'_1']) # creating a new column for differenced features
    df.drop(columns = [col,col+'_1'], inplace = True) # dropping two columns

final_df = df_optimized(df)
