import streamlit as st

import pandas as pd

import numpy as np

import pickle

# ----------------------------

# Load trained model, features, and scaler

# ----------------------------

model = pickle.load(open('models/churn_model_rf.pkl', 'rb'))

feature_cols = pickle.load(open('models/feature_columns.pkl', 'rb'))

scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# ----------------------------

# Streamlit app layout

# ----------------------------

st.title("📞 Telecom Customer Churn Prediction")

st.markdown("Predict whether a customer is likely to churn using their account information.")

st.sidebar.header("Customer Information")

# ----------------------------

# Function to get user input

# ----------------------------

def user_input_features():

    gender = st.sidebar.selectbox("Gender", ["Male","Female"])

    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0,1])

    partner = st.sidebar.selectbox("Has Partner?", ["Yes","No"])

    dependents = st.sidebar.selectbox("Has Dependents?", ["Yes","No"])

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

    phone_service = st.sidebar.selectbox("Phone Service", ["Yes","No"])

    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    internet_service = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    online_security = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])

    online_backup = st.sidebar.selectbox("Online Backup", ["Yes","No","No internet service"])

    device_protection = st.sidebar.selectbox("Device Protection", ["Yes","No","No internet service"])

    tech_support = st.sidebar.selectbox("Tech Support", ["Yes","No","No internet service"])

    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes","No","No internet service"])

    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month","One year","Two year"])

    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes","No"])

    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check","Mailed check",

                                                            "Bank transfer (automatic)","Credit card (automatic)"])

    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=1.0)

    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=1.0)

    # Map inputs to dataframe

    data = {

        'SeniorCitizen': senior_citizen,

        'tenure': tenure,

        'MonthlyCharges': monthly_charges,

        'TotalCharges': total_charges,

        'Partner': 1 if partner=="Yes" else 0,

        'Dependents': 1 if dependents=="Yes" else 0,

        'PhoneService': 1 if phone_service=="Yes" else 0,

        'MultipleLines': 0 if multiple_lines in ["No","No phone service"] else 1,

        'OnlineSecurity': 0 if online_security in ["No","No internet service"] else 1,

        'OnlineBackup': 0 if online_backup in ["No","No internet service"] else 1,

        'DeviceProtection': 0 if device_protection in ["No","No internet service"] else 1,

        'TechSupport': 0 if tech_support in ["No","No internet service"] else 1,

        'StreamingTV': 0 if streaming_tv in ["No","No internet service"] else 1,

        'StreamingMovies': 0 if streaming_movies in ["No","No internet service"] else 1,

        'PaperlessBilling': 1 if paperless_billing=="Yes" else 0,

    }

    # One-hot encoding for multi-class

    data['gender_Female'] = 1 if gender=="Female" else 0

    data['gender_Male'] = 1 if gender=="Male" else 0

    data['InternetService_DSL'] = 1 if internet_service=="DSL" else 0

    data['InternetService_Fiber optic'] = 1 if internet_service=="Fiber optic" else 0

    data['InternetService_No'] = 1 if internet_service=="No" else 0

    data['Contract_Month-to-month'] = 1 if contract=="Month-to-month" else 0

    data['Contract_One year'] = 1 if contract=="One year" else 0

    data['Contract_Two year'] = 1 if contract=="Two year" else 0

    data['PaymentMethod_Electronic check'] = 1 if payment_method=="Electronic check" else 0

    data['PaymentMethod_Mailed check'] = 1 if payment_method=="Mailed check" else 0

    data['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method=="Bank transfer (automatic)" else 0

    data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method=="Credit card (automatic)" else 0

    features = pd.DataFrame(data, index=[0])

    return features

# ----------------------------

# Get user input

# ----------------------------

input_df = user_input_features()

# Scale numeric columns using training scaler

num_cols = ['tenure','MonthlyCharges','TotalCharges']

input_df[num_cols] = scaler.transform(input_df[num_cols])

# Ensure all features exist and in same order

for col in feature_cols:

    if col not in input_df.columns:

        input_df[col] = 0

input_df = input_df[feature_cols]

# Display input

st.subheader("Customer Input")

st.write(input_df)

# ----------------------------

# Predict button

# ----------------------------

if st.button("Predict Churn"):

    prediction_proba = model.predict_proba(input_df)[:,1][0]

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")

    if prediction==1:

        st.success(f"The customer is likely to **churn**.\nProbability: {prediction_proba:.2f}")

    else:

        st.warning(f"The customer is **not likely to churn**.\nProbability: {prediction_proba:.2f}")
 