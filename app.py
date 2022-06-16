import streamlit as st
from backend import classify, classify_proba
import matplotlib.pyplot as plt
import time
import pandas as pd
import plotly.express as px
import numpy as np
import math

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

original_title = '''<p style="font-family:Courier; font-size: 50px;
font-weight:bold; text-align:center;"><span style="color:Blue;">DRUG</span><span style="color:Red;">-</span><span style="color:Green;">DRUG</span>
<span style="color:Black;">INTERACTION</span></p>'''
st.markdown(original_title, unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    drug1 = st.text_input("Input Drug 1", 'Aspirin')
with c2:
    drug2 = st.text_input('Input Drug 2', 'paracetamol')


drug = (f"{drug1} interact {drug2}")

col1, col2, col3 = st.columns(3)

with col1:
    pass
with col3:
    pass
with col2 :
    interact_button = st.button('Discover Side Effects')



if interact_button:

    with st.spinner(text="Calculating probabilities..."):

        pred = classify(drug1, drug2)
        proba = classify_proba(drug1, drug2)

        prediction = {}
        prediction["side_effects"] = []
        prediction["probability"] = proba
        prediction["severity"] = []

        for item in pred:
            prediction["severity"].append(item.split()[-1].replace("-", ""))
            prediction["side_effects"].append(" ".join(item.split()[:-1]).replace("-", ""))

        prediction_df = pd.DataFrame(prediction, columns = ["side_effects",
                                                        "probability",
                                                        "severity"])
        prediction_mild = prediction_df[prediction_df["severity"] == "Mild"]
        prediction_moderate = prediction_df[prediction_df["severity"] == "Moderate"]
        prediction_severe = prediction_df[prediction_df["severity"] == "Severe"]

        fig_mild = plt.figure(figsize = (10, 5))
        plt.barh(y = "side_effects", width = "probability", data = prediction_mild,
                 color = "green")
        plt.xlim([0, 1])
        plt.xlabel("Probability")
        plt.title("Mild Side Effects Probability")


        fig_moderate = plt.figure(figsize = (10, 5))
        plt.barh(y = "side_effects", width = "probability",
                 data = prediction_moderate, color = "orange")
        plt.xlim([0, 1])
        plt.xlabel("Probability")
        plt.title("Moderate Side Effects Probability")

        fig_severe = plt.figure(figsize = (10, 5))
        plt.barh(y = "side_effects", width = "probability",
                 data = prediction_severe, color = "maroon")
        plt.xlim([0, 1])
        plt.xlabel("Probability")
        plt.title("Severe Side Effects Probability")

    st.pyplot(fig_mild)
    st.pyplot(fig_moderate)
    st.pyplot(fig_severe)

st.write("Disclaimers:")
st.write('''1. This prediction is based on chemical structure of the drug.
         Physical properties of the drug itself is not taken into account when conducting
         the prediction.''')
st.write('''2. The prediction itself has 75% accuracy. As such, when using this
         model to make any prediction, user should be aware that there are
         uncertainties inherent in attempting to make such prediction, and thus
         should not be relying upon this model completely as medical advice.''')
st.write('''3. The model prediction does not capture the frequency of side effects
         occuring.''')
st.write('''4. The model itself does not take into account interaction of more than
         2 drugs. If user is taking more than two drugs concurrently, user
         should get medical advice from certified medical professionals.''')
st.write('''5. The model does not take into account the person’s medical history, which will
         affect the chances and frequency of side effects occuring.''')
