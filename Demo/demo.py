import pandas as pd
import streamlit as st
import numpy as np
import time
import joblib 
from sklearn import feature_selection, linear_model, model_selection, preprocessing
from sklearn import pipeline
from sklearn import model_selection
# from sklearn import feature_extraction
from sklearn import feature_selection
from sklearn import metrics
from datetime import datetime, timedelta

import tensorflow as tf

# Now you can use this loaded_model for inference in your production environment

# Load models
logistic_model = joblib.load('../Models/logistic_model.pkl')['classifier']  
logistic_vectorizer = joblib.load('../Models/logistic_vectorizer.pkl')
naive_bayes_model = joblib.load('../Models/naive_bayes_model.pkl')['classifier']  
naive_bayes_vectorizer = joblib.load('../Models/naive_bayes_vectorizer.pkl')
# random_forest_model = joblib.load('../Models/random_forest_model.pkl')['classifier'] 
# random_forest_vectorizer = joblib.load('../Models/random_forest_vectorizer.pkl')

def selectModel(selected_model):
        if selected_model == 'Logistic Regression':
            return logistic_model, logistic_vectorizer  
        elif selected_model == 'Naive Bayes':
            return naive_bayes_model, naive_bayes_vectorizer
        # elif selected_model == 'Random Forest':
        #     return random_forest_model, random_forest_vectorizer
        else:
            return 'Invalid model'

# Model prediction
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

    processed_text = vectorizer.transform([text_input])
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


class Home:
    def pageConfig():
        st.set_page_config(
            page_title='Text Classification Demo',
            page_icon=':newspaper:',
            layout='wide'
        )

    def createHeading():
        st.title('Text Classification Demo')
        st.subheader(f":red[{'For classifying newspaper headlines'}]")
        now = datetime.now().strftime('%H:%M')

# Configure webpage
Home.pageConfig()

title_column, model_column, results_column = st.columns(3)

with results_column:
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.markdown('\n\n')
    st.subheader('Results')
    st.markdown('\n\n')

with title_column:
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
        ('Logistic Regression', 'Naive Bayes')
    )

    st.subheader('Enter newspaper headline here')
    headline = st.text_input('Headline')

    if st.button('Run'):
        prediction, confidence, second_guess, second_confidence = modelPrediction(headline, selected_model)
        with results_column:
            st.write('**Predited newspaper section:** ', prediction)
            st.write('**Confidence:**', confidence)
            st.write('**Second Prediction:**', second_guess)
            st.write('**Confidence:**', second_confidence)
        
        
        
        


    