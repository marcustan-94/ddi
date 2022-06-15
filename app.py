import streamlit as st
from backend import classify, classify_proba
import matplotlib.pyplot as plt
import time
import pandas as pd
import plotly.express as px
import numpy as np
import math

original_title = '''<p style="font-family:Courier; color:Black; font-size: 50px;">DRUG-DRUG INTERACTION</p>'''
st.markdown(original_title, unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    drug1 = st.text_input('Input 1st Drug', 'Aspirin')
with c2:
    drug2 = st.text_input('Input 2nd Drug', 'paracetamol')

drug = (f"{drug1} interact {drug2}")

col1, col2, col3 = st.columns(3)

with col1:
    pass
with col3:
    pass
with col2 :
    interact_button = st.button('Discover Side Effects')



if interact_button:

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'State-Of-The-Art Advanced AI Model Computing... {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

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
    # prediction_df = prediction_df.sort_values(by = "probability")

    ##bar chart
    # fig = px.bar(prediction_df, x="probability", y="side_effects", color = "severity",
    #              orientation='h', color_discrete_map={"mild": "Green",
    #                                                   "moderate": "Orange",
    #                                                   "severe": "Maroon"})
    # st.plotly_chart(fig)

    #bubble
    X = []
    Y = []
    number_list =  [i for i in range(math.floor(len(pred)/5))]
    remainder_list = [i+1 for i in range(len(pred)%5)]
    for i in number_list:
        X_dummy = [1,2,3,4,5]
        X.extend(X_dummy)
        Y_dummy = [i*5, i*5, i*5, i*5, i*5]
        Y.extend(Y_dummy)
    while len(Y) < len(pred):
        Y.append(len(number_list)*5)
    X.extend(remainder_list)

    prediction_df["X"] = X
    prediction_df["Y"] = Y
    prediction_df["probability"] = prediction_df["probability"] * 100

    fig = px.scatter(prediction_df, x="X", y="Y", size="probability",
                     color = "severity", hover_name = "side_effects",
                     color_discrete_map={"Mild": "green",
                                         "Moderate": "orange",
                                         "Severe": "red"},
                     log_x=False, hover_data = {"X": False, "Y": False},
                     text = "probability",
                     title = "Drug-Drug Interaction Side Effects",
                     category_orders={"severity": ["Mild", "Moderate",
                                                   "Severe"]})
    fig.update_xaxes(visible = False, showticklabels=False, showgrid=False,
                     zeroline=False)
    fig.update_yaxes(visible = False, showticklabels=False, showgrid=False,
                     zeroline=False)
    fig.update_layout(title = {"x": 0.5, "y": 0.9}, legend=dict(orientation="h",
                                                                y=0, x=0.2))

    st.plotly_chart(fig, use_container_width=True)
