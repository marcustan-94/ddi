#!/usr/bin/env python
# coding: utf-8

# # Cleaning DataSet

# In[1]:


import pandas as pd
import re
import numpy as np


# ## Text File to DataFrame

# In[2]:


pattern = r"(\S*)\tDRUGSMIL\t(\S*)" #r"DRUGSMIL\t(/w+)"
text_to_search = 'D00AAN\tDRUGSMIL\tC1CCN2CCC3C(=CC(CCC=CC1)(C4C3(C2)CC5N4CCCC(=O)CC5)O)C6=NC=CC7=C6NC8=C7C=CC=C8OS(=O)(=O)C9=CC=C(C=C9)Cl'
re.findall(pattern, text_to_search)


# In[3]:


filepath = "../lewagon-ddi/raw_data/ttd_drug_ids.txt"
with open(filepath, encoding="utf-8") as f:
    drug_str = f.read()


# In[4]:


drug_list = drug_str.split("_______________________________________________________________________")
len(drug_list)


# In[5]:


actual_data = drug_list[2]


# In[6]:


drug_dict = {
    "DRUG__ID" :[],
    "DRUGSMIL":[]
}


# In[7]:


data2 = actual_data.split('\n')
data3 = list(filter(lambda a: a != '\t\t', data2))
data4 = list(filter(lambda a: a != '_\t\t', data3))


# In[8]:


for i in data4:
    pattern = r"(\S*)\tDRUGSMIL\t(\S*)"
    drugid = re.findall(pattern, i)
    if drugid !=[]:
        drug_id = drugid[0][0]
        drug_dict["DRUG__ID"].append(drug_id)
        drug_mil = drugid[0][1]
        drug_dict["DRUGSMIL"].append(drug_mil)


# In[9]:


len(drug_dict["DRUG__ID"])


# In[10]:


len(drug_dict["DRUGSMIL"])


# In[11]:


id_df = pd.DataFrame(drug_dict)


# In[12]:


id_df.columns = id_df.columns.str.replace('DRUG__ID', 'DrugID')


# ## TDD CSV to DataFrame

# In[13]:


df = pd.read_excel("../lewagon-ddi/raw_data/ttd_database.xlsx", index_col=0)


# In[14]:


df.reset_index(drop=False, inplace=True)


# In[15]:


tdd_df = df.drop(columns = 'Highest_status')
tdd_df.head()


# In[16]:


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


# In[17]:


tdd_df = tdd_df[tdd_df['MOA'].isin(drop_list) == False]
tdd_df['MOA'] = tdd_df['MOA'].replace(anta_list, 'Antagonist')
tdd_df['MOA'] = tdd_df['MOA'].replace(ago_list, 'Agonist')
tdd_df['MOA'] = tdd_df['MOA'].replace(mod_list, 'Modulator')
tdd_df['MOA'] = tdd_df['MOA'].replace(cart_list, 'CART')


# In[18]:


tdd_df['MOA'].unique()


# In[19]:


tdd_df.nunique()


# In[20]:


tdd_df['MOA'].value_counts()


# ## Concat 2 DrugID, and replace SMILE

# In[21]:


drug_df = pd.merge(tdd_df, id_df, on='DrugID', how='right')


# In[22]:


drug_df


# In[23]:


drug_df['DRUGSMIL'].isnull().sum()


# In[24]:


drug_df['TargetID'].isnull().sum()


# In[25]:


drug_df = drug_df.dropna()


# In[26]:


drug_df


# In[27]:


drug_df['DRUGSMIL'].nunique()


# ## TWOSIDE DataSet

# In[28]:


twoside_df = pd.read_csv("../lewagon-ddi/raw_data/twosides.csv", index_col=0)


# In[29]:


twoside_df.head()


# In[30]:


drug1_unique = twoside_df['Drug1'].unique()


# In[31]:


drug2_unique = twoside_df['Drug2'].unique()


# In[32]:


drug_unique = np.concatenate((drug1_unique, drug2_unique))


# In[33]:


drug_unique = sorted(set(drug_unique))


# In[34]:


drug_df2 = drug_df[drug_df['DRUGSMIL'].isin(drug_unique)]


# In[35]:


len(drug_df2['DRUGSMIL'].unique())


# In[36]:


len(drug_df2['DrugID'].unique())


# In[37]:


len(drug_df2['TargetID'].unique())


# In[38]:


drug_df2


# In[39]:


drug_df3 = drug_df2.pivot(columns=['TargetID'], values = 'MOA')


# In[40]:


drug_df4 = drug_df2.drop(columns = ['TargetID','MOA'])


# In[41]:


drug_df5 = pd.concat([drug_df4,drug_df3], axis=1)


# In[42]:


drug_df3


# In[43]:


drug_df6 = drug_df5.fillna('')
drug_df6 = drug_df6.groupby('DRUGSMIL').agg(lambda x: ''.join(x.unique()))


# In[44]:


drug_df6.reset_index(drop=False, inplace=True)


# In[45]:


drug_df7 = drug_df6.drop(columns = 'DrugID')


# In[46]:


drug_df7


# In[ ]:




