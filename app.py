import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load pre-trained Logistic Regression model and feature names
with open('logmodel.pkl', 'rb') as f:
    logmodel = pickle.load(f)

# Ensure feature alignment (replace with actual feature names from training)
feature_order = [
    'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Property_Area', 'Male', 'employed_Yes', 'married_Yes'
]

# Title of the app
st.title("Loan Approval Prediction App 💵")

# Description
st.write("""
This app predicts whether a loan application will be approved or not. 
Please fill in the details below and click on 'Predict Loan Status' to get the result.
""")

# User input form
st.sidebar.header('Input Features')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
married = st.sidebar.selectbox('Married', ['Yes', 'No'])
dependents = st.sidebar.selectbox('Number of Dependents', ['0', '1', '2', '3+'])
education = st.sidebar.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.sidebar.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.sidebar.number_input('Applicant Income', min_value=0, value=5000)
coapplicant_income = st.sidebar.number_input('Coapplicant Income', min_value=0, value=0)
loan_amount = st.sidebar.number_input('Loan Amount (in thousands)', min_value=0, value=100)
credit_history = st.sidebar.selectbox('Credit History', [1.0, 0.0])
property_area = st.sidebar.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Map inputs to match training features
input_data = {
    'Male': 1 if gender == 'Male' else 0,
    'married_Yes': 1 if married == 'Yes' else 0,
    'Dependents': int(dependents.replace('3+', '3')),
    'Education': 1 if education == 'Not Graduate' else 0,
    'employed_Yes': 1 if self_employed == 'Yes' else 0,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Credit_History': credit_history,
    'Property_Area': {'Urban': 0, 'Semiurban': 1, 'Rural': 2}[property_area],
}

# Convert input to DataFrame and align feature order
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_order]  # Align columns to match training

# Predict loan status using Logistic Regression model
if st.button('Predict Loan Status'):
    prediction = logmodel.predict(input_df)[0]

    # Display result
    if prediction == 1:
        st.success("Congratulations! Your loan application is likely to be approved.")
    else:
        st.error("Unfortunately, your loan application is unlikely to be approved.")
