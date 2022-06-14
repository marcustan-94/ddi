import streamlit as st
import time
import numpy as np
import pandas as pd
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt
import time
##Header for Streamlit Websitse##
original_title = '''<p style="font-family:Courier; color:Black; font-size: 50px;">DRUG-DRUG INTERACTION</p>'''
st.markdown(original_title, unsafe_allow_html=True)

##Split 2 Textbox side by side
c1, c2 = st.columns(2)
with c1:
    drug1 = st.text_input('Input 1st Drug', 'Aspirin')
with c2:
    drug2 = st.text_input('Input 2nd Drug', 'paracetamol')
drug = (f"{drug1} interact {drug2}")
col1, col2, col3 , col4, col5 = st.columns(5)

## Centering the clickbox
with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    interact_button = st.button('INTERACT!!!')


## Creating a Progress Bar After button is clicked!
if interact_button:
    f'{drug1} interacting {drug2}' ## Header for the progress bar(part of code)
    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'State-Of-The-Art Advanced AI Model to Computing... {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    'List of Side Effect' ## End_text_result for the progress bar(part of code)


# # Create some sample text
# text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'

# # Create and generate a word cloud image:
# wordcloud = WordCloud().generate(text)

# # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# st.pyplot()
