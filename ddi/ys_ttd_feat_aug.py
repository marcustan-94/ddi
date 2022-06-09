import pandas as pd
import re
import numpy as np

def tdd_df_moa_reclassification():
    '''
    Reclassifies MOA column of tdd_df into 5 classes: antagonist, agonist, modulator, cart, ligang
    '''
    tdd_df = pd.read_excel("../raw_data/ttd_database.xlsx")
    tdd_df = tdd_df.drop(columns='Highest_status')  # dropping unecessary columns

    anta_list = ['Inhibitor', 'Antagonist', 'Blocker', 'Inhibitor (gating inhibitor)','Blocker (channel blocker)',
                 'Disrupter', 'Suppressor', 'Inactivator','Inverse agonist','Inhibitor; Antagonist; Blocker', 'antagonist',
                 'Antagonist (gating inhibitor)','Antagonist (channel blocker)', 'Antagonist; Antagonist; Antagonist',
                 'Agonis; Antagonist','Agonis; Inverse agonist']
    ago_list = ['Agonist', 'Activator', 'Stimulator', 'Immunostimulant', 'Enhancer', 'Inducer', 'Regulator (upregulator)',
                'Cofactor', 'Partial agonist', 'Co-agonist', 'Stimulator ','Agonist ','Modulator (Agonist)']
    mod_list = ['Modulator', 'Binder', 'Binder (minor groove binder)', 'Immunomodulator', 'Modulator (allosteric modulator)',
                'Immunomodulator (Immunostimulant)','Regulator', 'Immunomodulator ','Modulator (minor groove binder)',
                'Modulator (upregulator)','Modulator ']
    cart_list = ['CAR-T-Cell-Therapy', 'CAR-T-Cell-Therapy(Dual specific)','CART(Dual specific)']
    drop_list = ['Chelator', 'Reactivator', 'Intercalator', 'Antisense', 'Immune response agent', 'Stabilizer','Stablizer',
                'Opener', 'Breaker', 'Degrader', 'Replacement', 'Antisense ','.']

    tdd_df = tdd_df[tdd_df['MOA'].isin(drop_list) == False]  # removing rows that have MOA in the drop_list
    tdd_df['MOA'] = tdd_df['MOA'].replace(anta_list, 'Antagonist')
    tdd_df['MOA'] = tdd_df['MOA'].replace(ago_list, 'Agonist')
    tdd_df['MOA'] = tdd_df['MOA'].replace(mod_list, 'Modulator')
    tdd_df['MOA'] = tdd_df['MOA'].replace(cart_list, 'CART')

    return tdd_df

def smiles_extractor():
    '''
    Extracting corresponding SMILES string of each drug ID from ttd_drug_ids.txt
    Creates a dataframe of 2 columns; column 1: Drug_ID, column 2: SMILES
    '''
    # loading text file into variable text
    filepath = "../raw_data/ttd_drug_ids.txt"
    with open(filepath, encoding="utf-8") as f:
        text = f.readlines()

    drug_dict = {
        "drug_id" :[],
        "smiles":[]
    }

    # pattern to search using regex, to pull out drug ID and corresponding SMILES string
    for row in text:
        pattern = r"(\S*)\tDRUGSMIL\t(\S*)"
        id_smiles = re.findall(pattern, row)
        if id_smiles !=[]:
            drug_dict["drug_id"].append(id_smiles[0][0])
            drug_dict["smiles"].append(id_smiles[0][1])

    drug_id_smiles_df = pd.DataFrame(drug_dict)

    # dropping all rows where the drug_id is different but the smiles is the same
    # drug_id_smiles_df.drop_duplicates(subset='smiles', keep=False, inplace=True)
    return drug_id_smiles_df

def twosides_unique_smiles_function():
    twosides_df = pd.read_csv("../raw_data/twosides.csv", index_col=0)
    twosides_unique_smiles = np.concatenate((twosides_df['Drug1'].unique(), twosides_df['Drug2'].unique()))
    twosides_unique_smiles = list(set(twosides_unique_smiles))  # removing duplicates
    return twosides_unique_smiles


if __name__ == '__main__':
    tdd_df = tdd_df_moa_reclassification()
    drug_id_smiles_df = smiles_extractor()
    tdd_df = pd.merge(tdd_df, drug_id_smiles_df, left_on='DrugID', right_on='drug_id').drop(columns='drug_id')
    twosides_unique_smiles = twosides_unique_smiles_function()
    tdd_df = tdd_df[tdd_df['smiles'].isin(twosides_unique_smiles)]  # filtering out only the rows of drugs
                                                                    # that are present in the twosides df

    # right now there are some rows in tdd_df that has the same smiles but different drug_id, the following
    # steps aim to remove these rows
    smiles_grouped_df = tdd_df.groupby('smiles', as_index=False).nunique()
    unique_smiles = smiles_grouped_df[smiles_grouped_df['DrugID'] == 1]['smiles']
    tdd_df = tdd_df[tdd_df['smiles'].isin(unique_smiles)]

    pivot_df = tdd_df.pivot(columns=['TargetID'], values='MOA')  # creating 313 columns, each column is a unique TargetID
    tdd_df = pd.concat([tdd_df['smiles'], pivot_df], axis=1)  # adding column for SMILES string
    tdd_df.fillna('', inplace=True)
    tdd_df = tdd_df.groupby('smiles', as_index=False).agg(lambda x: ''.join(x.unique()))
