import numpy as np
import pandas as pd
from ddi.utils import get_data_filepath, get_rawdata_filepath

# There are two datasets online that has worked on reclassification of side effects according to severity.
# The purpose of this .py file is to reference these two databases when reclassifying the side effects
# in the twosides dataset. The remaining side effects that are not found in these two datasets will then
# be classified manually based on our own judgement.

def score_category(score, gottlieb_or_lavertu):
    '''
    Reclassifying side effects into 3 classes; 1: mild, 2: moderate, 3: severe
    Slightly different thresholds are used for the Lavertu and Gottlieb datasets.
    These thresholds were decided by own judgement after looking through the datasets.
    '''
    if gottlieb_or_lavertu == 'lavertu':
        if score < 0.4:
            score = 1
        elif score < 0.8:
            score = 2
        elif score >= 0.8:
            score = 3
        else:
            score = 0
    elif gottlieb_or_lavertu == 'gottlieb':
        if score < 0.35:
            score = 1
        elif score < 0.7:
            score = 2
        elif score >= 0.7:
            score = 3
        else:
            score = 0
    return score


if __name__ == '__main__':
    # importing relevant datasets
    lavertu = pd.read_csv(get_rawdata_filepath('lavertu.csv'))
    gottlieb = pd.read_csv(get_rawdata_filepath('gottlieb.csv'))
    twosides_labels = pd.read_csv(get_data_filepath('sub_system_reclassification.csv'))

    # cleaning the lavertu and gottlieb dfs
    lavertu = lavertu.drop(columns=['cui', 'pt_code', 'saedr_score_std'])
    lavertu.columns = ['side_effect', 'score_lavertu']
    gottlieb = gottlieb.drop(columns='Rank Stdev (% out 2929)')
    gottlieb.columns = ['side_effect', 'score_gottlieb']
    gottlieb['side_effect'] = gottlieb['side_effect'].str.lower().str.replace('-', ' ')

    # merging the twosides side effects df with lavertu and gottlieb dfs
    merged_df = twosides_labels.merge(lavertu, on='side_effect', how='left')
    merged_df = merged_df.merge(gottlieb, on='side_effect', how='left')

    # categorising side effects into mild (1), moderate (2) and severe (3)
    merged_df['score_lavertu_cat'] = merged_df['score_lavertu'].apply(score_category, args=('lavertu',))
    merged_df['score_gottlieb_cat'] = merged_df['score_gottlieb'].apply(score_category, args=('gottlieb',))

    # consolidating lavertu and gottlieb scores
    merged_df['score_consolidated'] = 0
    for index, row in merged_df.iterrows():
        # if both lavertu and gottlieb had the same scoring, then just use that scoring
        if row['score_lavertu_cat'] == row['score_gottlieb_cat']:
            merged_df.loc[index, 'score_consolidated'] = row['score_lavertu_cat']
        # if only one of the two (lavertu and gottlieb) has scoring, then use the available scoring
        elif row['score_lavertu_cat'] == 0 or row['score_gottlieb_cat'] == 0:
            merged_df.loc[index, 'score_consolidated'] = (row['score_lavertu_cat'] + row['score_gottlieb_cat'])

    # after consolidation, rows where the consolidated score is 0 indicates that either the scoring information was
    # not available from both lavertu and gottlieb, or there was a discrepancy between the two scores
    # hence the merged df will be exported to csv for further manual reclassifcation
    merged_df.to_csv(get_data_filepath('partial_severity_reclassification.csv'), index=False)
