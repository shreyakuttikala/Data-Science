#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        log_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('train_columns.pkl', 'rb') as file:
        train_columns = pickle.load(file)
    st.success("Model, scaler, and feature columns loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model files not found. Please run the 'logistic_regression_model.py' script first to train and save the model.")
    st.stop() 


# In[3]:


st.title("🚢 Titanic Survival Predictor")
st.markdown("""
    This application predicts the survival probability of a passenger on the Titanic
    based on a Logistic Regression model.
    Fill in the details below to get a prediction!
""")


# In[4]:


st.header("Passenger Details")


# In[5]:


col1, col2 = st.columns(2)


# In[6]:


with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30, help="Age of the passenger")
    sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0, help="Number of siblings or spouses traveling with the passenger")


# In[7]:


with col2:
    parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0, help="Number of parents or children traveling with the passenger")
    fare = st.number_input("Fare", value=30.0, min_value=0.0, max_value=1000.0, help="Fare paid for the ticket")
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")


# In[8]:


def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, train_columns, scaler):
    # Create a dictionary from user inputs
    input_data = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Has_Cabin': 0, # Default to 0 as we don't have this input directly
        'FamilySize': sibsp + parch + 1,
        'IsAlone': 1 if (sibsp + parch) == 0 else 0,
        'Sex_male': 1 if sex == 'male' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0,
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensure all columns from training data are present, fill missing with 0
    # This is crucial for consistency with the trained model's features
    processed_input = input_df.reindex(columns=train_columns, fill_value=0)

    # Scale numerical features using the pre-fitted scaler
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])

    return processed_input


# In[9]:


if st.button("Predict Survival"):
    # Preprocess the user input
    processed_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, train_columns, scaler)

    # Make prediction
    prediction = log_model.predict(processed_input)
    prediction_proba = log_model.predict_proba(processed_input)[:, 1] # Probability of survival

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success(f"**The model predicts that this passenger would SURVIVE!**")
        st.balloons()
    else:
        st.error(f"**The model predicts that this passenger would NOT SURVIVE.**")

    st.info(f"Survival Probability: **{prediction_proba[0]:.2f}**")

    st.markdown("---")
    st.subheader("How to Interpret:")
    st.write(f"A survival probability closer to 1.0 indicates a higher chance of survival, while a probability closer to 0.0 indicates a lower chance.")
    st.write("This prediction is based on the Logistic Regression model trained on the Titanic dataset.")


# In[10]:


st.markdown("---")
st.markdown("Developed by Gemini for your Logistic Regression Assignment.")


# In[ ]:




