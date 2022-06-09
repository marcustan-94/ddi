import pandas as pd
import re
import numpy as np


pattern = r"(\S*)\tDRUGSMIL\t(\S*)" #r"DRUGSMIL\t(/w+)"
text_to_search = 'D00AAN\tDRUGSMIL\tC1CCN2CCC3C(=CC(CCC=CC1)(C4C3(C2)CC5N4CCCC(=O)CC5)O)C6=NC=CC7=C6NC8=C7C=CC=C8OS(=O)(=O)C9=CC=C(C=C9)Cl'
re.findall(pattern, text_to_search)


filepath = "../raw_data/ttd_drug_ids.txt"
with open(filepath, encoding="utf-8") as f:
    drug_str = f.read()

drug_list = drug_str.split("_______________________________________________________________________")

actual_data = drug_list[2]


drug_dict = {
    "DRUG__ID" :[],
    "DRUGSMIL":[]
}


data2 = actual_data.split('\n')
data3 = list(filter(lambda a: a != '\t\t', data2))
data4 = list(filter(lambda a: a != '_\t\t', data3))

for i in data4:
    pattern = r"(\S*)\tDRUGSMIL\t(\S*)"
    drugid = re.findall(pattern, i)
    if drugid !=[]:
        drug_id = drugid[0][0]
        drug_dict["DRUG__ID"].append(drug_id)
        drug_mil = drugid[0][1]
        drug_dict["DRUGSMIL"].append(drug_mil)


id_df = pd.DataFrame(drug_dict)


id_df.columns = id_df.columns.str.replace('DRUG__ID', 'DrugID')

df = pd.read_excel("../lewagon-ddi/raw_data/ttd_database.xlsx", index_col=0)

df.reset_index(drop=False, inplace=True)


tdd_df = df.drop(columns = 'Highest_status')
tdd_df.head()


anta_list = ['Inhibitor', 'Antagonist', 'Blocker', 'Inhibitor (gating inhibitor)',
             'Blocker (channel blocker)', 'Disrupter', 'Suppressor', 'Inactivator', 'Inverse agonist',
             'Inhibitor; Antagonist; Blocker', 'antagonist', 'Antagonist (gating inhibitor)','Antagonist (channel blocker)',
            'Antagonist; Antagonist; Antagonist','Agonis; Antagonist','Agonis; Inverse agonist']
ago_list = ['Agonist', 'Activator', 'Stimulator', 'Immunostimulant', 'Enhancer', 'Inducer', 'Regulator (upregulator)',
            'Cofactor', 'Partial agonist', 'Co-agonist', 'Stimulator ','Agonist ','Modulator (Agonist)']
mod_list = ['Modulator', 'Binder', 'Binder (minor groove binder)', 'Immunomodulator', 'Modulator (allosteric modulator)',
            'Immunomodulator (Immunostimulant)','Regulator', 'Immunomodulator ','Modulator (minor groove binder)',
           'Modulator (upregulator)','Modulator ']
cart_list = ['CAR-T-Cell-Therapy', 'CAR-T-Cell-Therapy(Dual specific)','CART(Dual specific)']
drop_list = ['Chelator', 'Reactivator', 'Intercalator', 'Antisense', 'Immune response agent', 'Stabilizer','Stablizer',
             'Opener', 'Breaker', 'Degrader', 'Replacement', 'Antisense ','.']

tdd_df = tdd_df[tdd_df['MOA'].isin(drop_list) == False]
tdd_df['MOA'] = tdd_df['MOA'].replace(anta_list, 'Antagonist')
tdd_df['MOA'] = tdd_df['MOA'].replace(ago_list, 'Agonist')
tdd_df['MOA'] = tdd_df['MOA'].replace(mod_list, 'Modulator')
tdd_df['MOA'] = tdd_df['MOA'].replace(cart_list, 'CART')




# ## Concat 2 DrugID, and replace SMILE



drug_df = pd.merge(tdd_df, id_df, on='DrugID', how='right')
drug_df = drug_df.dropna()



twoside_df = pd.read_csv("../lewagon-ddi/raw_data/twosides.csv", index_col=0)
drug1_unique = twoside_df['Drug1'].unique()
drug2_unique = twoside_df['Drug2'].unique()

drug_unique = np.concatenate((drug1_unique, drug2_unique))



drug_unique = sorted(set(drug_unique))


drug_df2 = drug_df[drug_df['DRUGSMIL'].isin(drug_unique)]
drug_df3 = drug_df2.pivot(columns=['TargetID'], values = 'MOA')
drug_df4 = drug_df2.drop(columns = ['TargetID','MOA'])
drug_df5 = pd.concat([drug_df4,drug_df3], axis=1)
drug_df6 = drug_df5.fillna('')
drug_df6 = drug_df6.groupby('DRUGSMIL').agg(lambda x: ''.join(x.unique()))
drug_df6.reset_index(drop=False, inplace=True)
drug_df7 = drug_df6.drop(columns = 'DrugID')
