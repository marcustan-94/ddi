#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all needed modules
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem


# In[2]:


#opening twosides file as df
twosides_df = pd.read_csv("twosides.csv").drop(columns = "Unnamed: 0")


# In[3]:


#getting the list for drug1
drug_list_for_drug_1 = list(twosides_df["Drug1"].unique())


# In[4]:


#initiating mordred calculator
calc = Calculator(descriptors, ignore_3D=True)


# In[5]:


#building drug features for drug1
mols_for_drug_1 = [Chem.MolFromSmiles(item) for item in drug_list_for_drug_1]
drug1_features = calc.pandas(mols_for_drug_1)


# In[6]:


#drug1_features dataframe lacks the SMILE for drug1 itself so add in the SMILE
drug1_features["Drug1"] = drug_list_for_drug_1


# In[7]:


#shifting SMILE structure for drug1 to the front so it is easier to visualize
drug1_features = drug1_features.set_index("Drug1").reset_index()


# In[8]:


#building the list of columns to drop by selecting dtypes = "object", as we only want the float/int dtypes
#the list doesn't include "Drug1" column as we need the SMILE structures for merging
drug1_features_object_list = list(drug1_features.select_dtypes("object").keys()[1:])


# In[9]:


#dropping the columns where dtypes = "objects", except the "Drug1" column where it contains the SMILE structure
drug1_features = drug1_features.drop(columns = drug1_features_object_list)


# In[10]:


#visualizing built drug1 features
drug1_features


# In[11]:


#converting the file to csv
drug1_features.to_csv("cxe_feat_eng_drug1_droppedcolumns.csv")


# In[12]:


#getting the list for drug2
drug_list_for_drug_2 = list(twosides_df["Drug2"].unique())


# In[13]:


#building drug features for drug2
mols_for_drug_2 = [Chem.MolFromSmiles(item) for item in drug_list_for_drug_2]
drug2_features = calc.pandas(mols_for_drug_2)


# In[14]:


#drug2_features dataframe lacks the SMILE for drug1 itself so add in the SMILE
drug2_features["Drug2"] = drug_list_for_drug_2


# In[15]:


#shifting SMILE structure for drug1 to the front so it is easier to visualize
drug2_features = drug2_features.set_index("Drug2").reset_index()


# In[16]:


#building the list of columns to drop by selecting dtypes = "object", as we only want the float/int dtypes
#the list doesn't include "Drug1" column as we need the SMILE structures for merging
drug2_features_object_list = list(drug2_features.select_dtypes("object").keys()[1:])


# In[17]:


#dropping the columns where dtypes = "objects", except the "Drug1" column where it contains the SMILE structure
drug2_features = drug2_features.drop(columns = drug2_features_object_list)


# In[18]:


#visualizing built drug1 features
drug2_features


# In[19]:


#converting the file to csv
drug2_features.to_csv("cxe_feat_eng_drug2_droppedcolumns.csv")

