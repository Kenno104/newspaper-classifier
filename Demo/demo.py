# -*- coding: utf-8 -*-
"""
Updated on Tue Oct 17 

@author: conkennedy

This functionality is demonstrated via Streamlit (https://streamlit.io/). A Python library that allows for the creation of web apps for demo purposes.
"""

import pandas as pd
import numpy as np
import streamlit as st
import joblib 
from datetime import datetime, timedelta
from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_hub as hub

from preprocessing import Headline

#NN model architecture - Commented out until model issues fixed
# def create_model(): 
#     model = "https://tfhub.dev/google/nnlm-en-dim50/2"
#     hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
#     model = keras.Sequential()
#     model.add(hub_layer)
#     model.add(keras.layers.Dense(16, activation='relu'))
#     model.add(keras.layers.Dense(16, activation='relu'))
#     model.add(keras.layers.Dense(4, activation='softmax'))
#     return model

# Load models
# NOTE: When developing - you may have to change the below paths to the models, i.e. use '../Models/logistic_model.pkl' if using Mac.
logistic_model = joblib.load('../Models/logistic_model.pkl')['classifier']  
logistic_vectorizer = joblib.load('../Models/logistic_vectorizer.pkl')
naive_bayes_model = joblib.load('../Models/naive_bayes_model.pkl')['classifier']  
naive_bayes_vectorizer = joblib.load('../Models/naive_bayes_vectorizer.pkl')
oneVsRest_model = joblib.load('../Models/oneVsRest_model.pkl')['classifier']
oneVsRest_vectorizer = joblib.load('../Models/oneVsRest_vectorizer.pkl')

# Neural nets - TensorFlow
# simple_nn_model = create_model()
# simple_nn_model.load_weights('../Models/simple_NN.h5')

# Branding
logo = Image.open('../Demo/Branding/deloitte-logo.png')

# FUNCTIONS
# Basic page setup
class Home:
    def pageConfig():
        st.set_page_config(
            page_title='Text Classification Demo',
            page_icon=':newspaper:',
            layout='wide'
        )
        
        # Custom CSS for light theme
        light_theme_css = """
        <style>
        body {
            background-color: #FFFFFF;
        }
        </style>
        """
        # Inject custom CSS 
        st.markdown(light_theme_css, unsafe_allow_html=True)

    def createHeading():
        st.title('Text Classification Demo')
        st.markdown('<h3 style="color: #86BC25;">For classifying newspaper headlines</h3>', unsafe_allow_html=True)
        now = datetime.now().strftime('%H:%M')

# Load correct model & vectoriser
def selectModel(selected_model):
        if selected_model == 'Logistic Regression':
            return logistic_model, logistic_vectorizer  
        elif selected_model == 'Naive Bayes':
            return naive_bayes_model, naive_bayes_vectorizer
        elif selected_model == 'One Vs Rest':
            return oneVsRest_model, oneVsRest_vectorizer
        # elif selected_model == 'Simple Neural Network':
            # return simple_nn_model, None
        else:
            return 'Invalid model'

# Model prediction flow
def modelPrediction(text_input, selected_model):
    #Load correct model & vectoriser
    loaded_model, vectorizer = selectModel(selected_model)

    # Dictionary to map class index to class name
    class_dict = {
        'b': 'Business',
        't': 'Science and Technology',
        'e': 'Entertainment',
        'm': 'Health'
    }

    # Determine if using neural net or alternative model - Commented out until model issues fixed
    # if selected_model == 'Simple Neural Network':
    #     #CODE HERE
    # else:
    if vectorizer is not None:
        processed_text = vectorizer.transform([text_input])
    else:
        processed_text = [text_input]

    predicted_probabilities = loaded_model.predict_proba(processed_text)
    
    # Get the index of the most confident prediction
    predicted_class_index = predicted_probabilities.argmax()
    
    # Get the probability/confidence of the most confident prediction
    max_confidence = (predicted_probabilities.max() * 100).round(2)
    max_confidence = str(max_confidence) + '%'

    # Get the label of the most confident prediction
    predicted_class = loaded_model.classes_[predicted_class_index]
    prediction = class_dict.get(predicted_class, 'Invalid class')
    
    # Sort the probabilities in descending order
    sorted_indices = np.argsort(predicted_probabilities.ravel())[::-1]

    # Get the index of the second most confident prediction
    second_max_index = sorted_indices[1]
    
    # Confidence of second prediction
    second_max_confidence = predicted_probabilities.ravel()[second_max_index]
    second_max_confidence = str((second_max_confidence * 100).round(2)) + '%'

    # Get the label of the second most confident prediction
    second_predicted_class = loaded_model.classes_[second_max_index]
    second_prediction = class_dict.get(second_predicted_class, 'Invalid class')
    
    return prediction, max_confidence, second_prediction, second_max_confidence

# Configure webpage
Home.pageConfig()

# Page split into 3 columns (Info, Input, Output)
title_column, model_column, results_column = st.columns(3)

# Initalise Results column
with results_column:
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.subheader('Results')
    st.markdown('\n\n')

with title_column:
    st.image(logo, caption=None, width=200, use_column_width=False, clamp=False, channels='RGB', output_format='auto')
    Home.createHeading()
    text = """
    This is a text classification demo to classify newspaper headlines into one of four categories: 

    * Business
    * Science and Technology
    * Entertainment
    * Health

    The available models were trained on a BBC News dataset, which can be found [here](https://www.kaggle.com/c/learn-ai-bbc/data).

    **Instructions**

    Please select the model you would like to use for text classification and then enter a newspaper headline in the text box. Click the 'Run' button to classify the headline. The results will appear in the 'Results' column on the right.
    """
    st.markdown('\n\n')
    st.markdown(text)

with model_column:
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')

    st.subheader('Model')
    selected_model = st.selectbox(
        'Select model',
        ('Logistic Regression', 'Naive Bayes', 'One Vs Rest')
    )

    st.subheader('Enter newspaper headline here')
    headline = st.text_input('Headline')

    # Custom CSS for button hover color
    st.markdown(
        """
        <style>
        .stButton>button:hover {
            border-color: #86BC25;
            color: #86BC25;
        }
        .stButton>button:active {
        background-color: #86BC25;
        border-color: #86BC25;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button('Run'):
        # First, preprocess the input text (headline)
        # headline_instance = Headline(headline)
        # headline = headline_instance.process()
        
        # Now, give to model to infer result
        prediction, confidence, second_guess, second_confidence = modelPrediction(headline, selected_model)
        with results_column:
            st.write('**Predited newspaper section:** ', prediction)
            st.write('**Confidence:**', confidence)
            st.write('**Second Prediction:**', second_guess)
            st.write('**Confidence:**', second_confidence)
        

        
        
        


    