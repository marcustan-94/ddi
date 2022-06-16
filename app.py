import streamlit as st
from ddi.utils import get_rawdata_filepath
from backend import classify, load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


@st.cache(allow_output_mutation=True)
def loading_model():
    pipeline = load_model()
    return pipeline

pipeline = loading_model()

# loading probability dataset
probability_df = pd.read_csv(get_rawdata_filepath("sub_system_severity.csv"))
probability_df = probability_df.set_index("sub_system_severity")
probability_df = probability_df.rename(columns = {"severity": "probability %", "Side effects": "Location of side effects"})
probability_df["probability %"] = probability_df["probability %"] * 100
round_x = lambda x: round(x, 2)
probability_df["probability %"] = probability_df["probability %"].apply(round_x)

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
drug1 = c1.text_input("Input Drug 1", '')
drug2 = c2.text_input('Input Drug 2', '')

# inserting interact button
col1, col2, col3 = st.columns(3)
interact_button = col2.button('Discover Side Effects')

if interact_button:

    # inserting spinner for aesthetic purpose
    with st.spinner(text="Discovering possible side effects..."):

        # calling prediction from backend
        pred = classify(drug1, drug2, pipeline)

        #building dataframe from predictions for plot purposes
        prediction = {"Location of side effects": [],
                      "Severity": []}

        for item in pred:
            prediction["Location of side effects"].append(" ".join(item.split()[:-1]).replace("-", ""))
            prediction["Severity"].append(item.split()[-1].replace("-", "").capitalize())

        prediction_df = pd.DataFrame(prediction)
        prediction_df = prediction_df.assign(hack='').set_index('hack')
        print(prediction_df.to_string(index=False))


        severity_df = pd.read_csv(get_rawdata_filepath("severity.csv")).drop(columns = "Unnamed: 0")
        severity_df = severity_df[severity_df["severity"] != "zzz_delete_0"]
        severity = plt.figure(figsize = (10, 5), facecolor='#DCF1EE')
        plt.barh(y = "severity", width = "Y_cat", data = severity_df, color =
                 ["maroon", "green", "orange"])
        plt.xlim([0, 1])
        plt.xlabel("Probability")
        plt.title("Probability of Side Effects According to Severity")
        ax = plt.gca()
        ax.set_facecolor('#DCF1EE')

        st.pyplot(severity)

# writing disclaimers for user awareness
st.write('''
        Disclaimer:

        This model is purely intended for providing awareness on the possible side
        effects between drug pairs, and should not be used as a substitute for opinion from
        a certified medical professional. When in doubt, please always consult your doctor.
        ''')
