import joblib
import pandas as pd
import streamlit as st

from backend import load_model, get_smiles, classify
from ddi.utils import get_data_filepath


# Loading the model beforehand so that the user do not need to wait for the model
# to load when they click the "Discover" button. This saves waiting time.

model = load_model()


# Background image is inserted using CSS code
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

# Title of the website is inserted using HTML code
title = '''
<p style="font-family:Courier; font-size: 50px;font-weight:bold; text-align:center;">
<span style="color:Blue;">DRUG</span><span style="color:Red;">-</span><span style="color:Green;">DRUG</span>
<span style="color:Black;">INTERACTION</span></p>
'''
st.write(title, unsafe_allow_html=True)

# Input boxes for Drug A and Drug B are inserted side by side in one row
c1, c2 = st.columns(2)
drug1 = c1.text_input("Input Drug A", "")
drug2 = c2.text_input("Input Drug B", "")

# "Discover side Effects" button is inserted in the middle of the next row
col1, col2, col3 = st.columns(3)
interact_button = col2.button('Discover Side Effects')

if interact_button:

    # Spinner is inserted for aesthetic purpose
    with st.spinner(text="Discovering possible side effects..."):

        # If any of the input drugs cannot be converted into smiles structure,
        # an error message will be printed
        if 'Unable to find the drug, please try again.' in get_smiles(drug1, drug2):
            st.write('Unable to find the drug, please try again.')

        # If both drugs are successfully converted into smiles structures by the API,
        # prediction will be called from backend.py
        else:
            pred = classify(drug1, drug2, model)

            # Creates an empty dataframe containing an index column containing location of side effects
            # and 3 other columns: Mild, Moderate, Severe
            reclass_df = pd.read_csv(get_data_filepath('complete_severity_reclassification.csv'))
            reclass_df = reclass_df[(reclass_df['sub_system_severity']!='zzz_delete_0')]
            reclass_df['sub_system'] = reclass_df['sub_system_severity'].apply(lambda x: x[:x.find('-') - 1])
            df = pd.DataFrame(columns=['Mild', 'Moderate', 'Severe'],
                              index=reclass_df['sub_system'].unique(),
                              data=0)
            # Adding the value 1 to only the cells with the corresponding side effect location and severity
            for col in df:
                for row in df.index:
                    if pred.count(row + ' -' + col) != 0:
                        df.loc[row, col] = 1

            # Resetting index and sorting the dataframe according to the alphabetical order of side effect locations
            df = df.reset_index().rename(columns={'index': 'Location of side effects'})
            df = df.sort_values('Location of side effects')

            # Hide table index
            hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            # The severity columns of the dataframe are highlighted using colors
            # Mild: Green, Moderate: Orange, Severe: Red
            # Highlighting is only done if the cell contains value 1
            def mild_color(val):
                color = '#A2D9A4' if val==1 else 'white'
                return f'background-color: {color}; color: {color}'

            def moderate_color(val):
                color = '#FDD783' if val==1 else 'white'
                return f'background-color: {color}; color: {color}'

            def severe_color(val):
                color = '#DB5C4A' if val==1 else 'white'
                return f'background-color: {color}; color: {color}'

            # The dataframe is displayed and styled using css
            st.table(df.style.applymap(mild_color, subset=['Mild']).\
                applymap(moderate_color, subset=['Moderate']).\
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

# Disclaimer is written for user awareness
st.markdown('''
        **Disclaimer:**

        **This model is purely intended for providing awareness on the possible side
        effects between drug pairs, and should not be used as a substitute for opinion from
        a certified medical professional. When in doubt, please always consult your doctor.**
        ''')
