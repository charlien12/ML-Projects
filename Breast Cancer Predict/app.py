import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/breast_cancer_model.joblib")
scaler = joblib.load("models/breast_cancer_scaler.joblib")

# Title
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter the cell characteristics below to predict whether the tumor is **Benign (non-cancerous)** or **Malignant (cancerous)**.")

# Selected 10 features
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
]

# Create input fields for each feature
input_data = []
col1, col2 = st.columns(2)
for i, feature in enumerate(selected_features):
    if i % 2 == 0:
        val = col1.number_input(f"{feature}", min_value=0.0, value=10.0, step=0.1)
    else:
        val = col2.number_input(f"{feature}", min_value=0.0, value=10.0, step=0.1)
    input_data.append(val)

# Convert input into numpy array
input_array = np.array([input_data])

# Scale input
input_scaled = scaler.transform(input_array)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled)) * 100

    if prediction == 0:
        st.error(f"⚠️ Prediction: Malignant (Cancerous)\n\nConfidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Prediction: Benign (Non-cancerous)\n\nConfidence: {confidence:.2f}%")
