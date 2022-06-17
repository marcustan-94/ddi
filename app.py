import streamlit as st
from ddi.utils import get_rawdata_filepath, get_data_filepath
from backend import classify, load_model, get_smiles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


@st.cache(allow_output_mutation=True)
def loading_model():
    pipeline = load_model()
    return pipeline

pipeline = loading_model()


# inserting background image
CSS = """
h1 {
    color: red;
}
.stApp {
    background-image: url(https://img.freepik.com/free-vector/clean-medical-background_53876-116877.jpg?t=st=1655270812~exp=1655271412~hmac=c9b2645a1b14f347f98cfabef14363ee5203d2a941c0085ad3bdd11092eaf2af&w=996);
    background-size: cover;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# writing title
original_title = '''<p style="font-family:Courier; font-size: 50px;
font-weight:bold; text-align:center;"><span style="color:Blue;">DRUG</span><span style="color:Red;">-</span><span style="color:Green;">DRUG</span>
<span style="color:Black;">INTERACTION</span></p>'''
st.markdown(original_title, unsafe_allow_html=True)

# inserting input box for drug1 and drug2
c1, c2 = st.columns(2)
drug1 = c1.text_input("Input Drug A", '')
drug2 = c2.text_input("Input Drug B", '')

checkbox = """
    <style type="text/css">
    .st-ce {background-color: white;}
    </style>
    """
st.markdown(checkbox, unsafe_allow_html=True)

# inserting interact button
col1, col2, col3 = st.columns(3)
interact_button = col2.button('Discover Side Effects')


if interact_button:

    # inserting spinner for aesthetic purpose
    with st.spinner(text="Discovering possible side effects..."):

        if 'Unable to find the drug, please try again.' in get_smiles(drug1, drug2):
            st.write('Unable to find the drug, please try again.')

        # calling prediction from backend

        else:
            pred = classify(drug1, drug2, pipeline)

            reclass_df = pd.read_csv(get_data_filepath('complete_severity_reclassification.csv'))
            reclass_df = reclass_df[(reclass_df['sub_system_severity']!='zzz_delete_0') & \
                                    (reclass_df['sub_system_severity']!='Delete -Moderate')]
            reclass_df['sub_system'] = reclass_df['sub_system_severity'].apply(lambda x: x[:x.find('-') - 1])

            df = pd.DataFrame(columns=['Mild', 'Moderate', 'Severe'], index=reclass_df['sub_system'].unique(), data=0)

            for col in df:
                for row in df.index:
                    if pred.count(row + ' -' + col) != 0:
                        df.loc[row, col] = 1
            df = df.reset_index()
            df = df.rename(columns={'index': 'Location of side effects'})

            df.iloc[0, 0] = 'Bile duct / Gallbladder / Pancreas'
            df.iloc[1, 0] = 'Bladder / Urethra'
            df.iloc[4, 0] = 'Blood Electrolytes / Vitamins'
            df.iloc[7, 0] = 'Brain / Spinal Cord'
            df.iloc[9, 0] = 'General Respiratory System'
            df.iloc[13, 0] = 'General Gastrointestinal System'
            df.iloc[14, 0] = 'Hair / Nails'
            df.iloc[17, 0] = 'Joints / Connective Tissues'
            df.iloc[23, 0] = 'Muscles / Tendons'
            df.iloc[26, 0] = 'Oesophagus / Stomach'
            df.iloc[27, 0] = 'Oral Cavity / Salivary Glands / Throat'
            df.iloc[29, 0] = 'Rectum / Anus'
            df.iloc[33, 0] = 'Skin / Subcutaneous Fat'
            df.iloc[34, 0] = 'Small / Large Intestine'
            df.iloc[35, 0] = 'General Immune System'
            df.iloc[36, 0] = 'Thyroid / Parathyroid / Pituitary Gland'
            df.iloc[37, 0] = 'General Urinary System'

            df = df[(df['Location of side effects'] != 'Hair') & (df['Location of side effects'] != 'Multiple Systemic General')]
            df = df.sort_values('Location of side effects')

            hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            def mild_color(val):
                color = '#A2D9A4' if val==1 else 'white'
                return f'background-color: {color}; color: {color}'

            def moderate_color(val):
                color = '#FDD783' if val==1 else 'white'
                return f'background-color: {color}; color: {color}'

            def severe_color(val):
                color = '#DB5C4A' if val==1 else 'white'
                return f'background-color: {color}; color: {color}'

            # st.table(df)
            # st.table(df.style.set_properties(**{'background-color': 'black', 'color': 'green'}))
            st.table(df.style.applymap(mild_color, subset=['Mild']).\
                applymap(moderate_color, subset=['Moderate']).
                applymap(severe_color, subset=['Severe']))

            df_table = """
                <style type="text/css">
                .css-a51556 {font-weight: 900; color:black;}
                th {background-color: white; }
                td {width: 80px; white-space: nowrap;}
                table {background-color:white; color:white; }
                thead, tbody {border-width: 1.7px; border-color: black;}
                </style>
                """
            st.markdown(df_table, unsafe_allow_html=True)

# writing disclaimers for user awareness
st.markdown('''
        **Disclaimer:**

        **This model is purely intended for providing awareness on the possible side
        effects between drug pairs, and should not be used as a substitute for opinion from
        a certified medical professional. When in doubt, please always consult your doctor.**
        ''')
