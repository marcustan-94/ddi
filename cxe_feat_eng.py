#importing all needed modules
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

class RawData:
    def __init__(self, file_name):
        self.file_name = file_name

    def get_data(self):
        '''opening twosides file as df'''
        twosides_df = pd.read_csv(self.file_name).drop(columns = "Unnamed: 0")
        return twosides_df

    def get_list_for_drug(self, Drug_n):
        '''getting the list for Drug_n'''
        '''Drug_n should be in this format: Drug1, or Drug2.
        eg self.get_list_for_drug(Drug1)'''
        twosides_df = self.get_data()
        drug_list_for_drug = list(twosides_df[f"{Drug_n}"].unique())
        return drug_list_for_drug

    def build_features(self, Drug_n):
        '''initiating mordred calculator'''
        calc = Calculator(descriptors, ignore_3D=True)
        '''gettiing drug list'''
        drug_list_for_drug = self.get_list_for_drug(Drug_n)
        '''getting molecular data from items in drug_list_for_drug'''
        mols = [Chem.MolFromSmiles(item) for item in drug_list_for_drug]
        '''building pandas dataframe'''
        drug_features = calc.pandas(mols)
        '''drug_features dataframe lacks the SMILE for Drug_n itself so add in
        the SMILE'''
        drug_features[f"{Drug_n}"] = drug_list_for_drug
        '''shifting SMILE structure for Drug_n to the front so it is easier to
        visualize'''
        drug_features = drug_features.set_index(f"{Drug_n}").reset_index()
        '''building the list of columns to drop by selecting dtypes = "object",
        as we only want the float/int dtypes'''
        '''the list doesn't include "Drug_n" column as we need the SMILE
        structures for merging'''
        drug_features_object_list = list(drug_features.
                                          select_dtypes("object").keys()[1:])
        '''dropping the columns where dtypes = "objects", except the "Drug_n"
        column where it contains the SMILE structure'''
        drug_features = drug_features.drop(columns =
                                             drug_features_object_list)
        return drug_features

    def convert_to_csv(self, Drug_n, csv_name):
        '''converting the file to csv'''
        drug_features_df = self.build_features(Drug_n)
        drug_features_df.to_csv(csv_name)
        return "Conversion successful"
