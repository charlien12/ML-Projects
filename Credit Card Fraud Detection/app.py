# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------------
# Load model and scalers
# -------------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/time_scaler.pkl", "rb") as f:
    time_scaler = pickle.load(f)

with open("models/amount_scaler.pkl", "rb") as f:
    amount_scaler = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it's **Fraudulent** or **Legitimate**.")

st.sidebar.header("Transaction Input Features")

# -------------------------------
# User Input
# -------------------------------
def user_input_features():
    time = st.sidebar.number_input("Time (seconds)", 0, 100000, 0)
    amount = st.sidebar.number_input("Amount ($)", 0.0, 50000.0, 100.0)
    
    V_features = {}
    for i in range(1, 29):
        V_features[f"V{i}"] = st.sidebar.number_input(f"V{i}", -50.0, 50.0, 0.0)
    
    data = {'Time': time, 'Amount': amount}
    data.update(V_features)
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -------------------------------
# Scale Time & Amount
# -------------------------------
input_df['Time'] = time_scaler.transform(input_df[['Time']])
input_df['Amount'] = amount_scaler.transform(input_df[['Amount']])

# -------------------------------
# Match Feature Order
# -------------------------------
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("🔍 Prediction Result")
st.success("✅ Legitimate Transaction" if prediction[0] == 0 else "🚨 Fraudulent Transaction")

st.subheader("📊 Prediction Probability")
st.write(f"Legitimate: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Fraudulent: {prediction_proba[0][1]*100:.2f}%")

# -------------------------------
# Optional Charts
# -------------------------------
if st.checkbox("Show Example Test Data Confusion Matrix"):
    try:
        test_data = pd.read_csv("test_predictions.csv")  # optional
        cm = confusion_matrix(test_data['y_test'], test_data['y_pred'])
        st.subheader("Confusion Matrix Heatmap")
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate','Fraud'],
                    yticklabels=['Legitimate','Fraud'])
        st.pyplot(plt)
    except FileNotFoundError:
        st.warning("⚠️ test_predictions.csv not found.")
