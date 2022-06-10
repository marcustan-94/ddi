from operator import index
from ddi.utils import get_data_filepath, get_rawdata_filepath
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def reclassification_encoder(filename):
    reclass = pd.read_csv(get_rawdata_filepath('mt_reclassified_twosides_labels.csv'), index_col=0)
    reclass = reclass.dropna().drop(columns ='side_effect')

    le = LabelEncoder()
    le.fit(reclass['sub_system'])
    reclass['Y_cat']= le.transform(reclass['sub_system'])


    reclass.to_csv(get_data_filepath(filename))
    print(f'{filename} successfully created and stored in data folder')

if __name__ == '__main__':
    reclassification_encoder('gl_reclassification_encoded.csv')
