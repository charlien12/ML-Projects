# ---------------------------------------------------------
# Loan Approval Prediction App using Streamlit + RandomForest
# ---------------------------------------------------------

import streamlit as st
import joblib
import numpy as np

# ---------------------------------------------------------
# Load the saved model using joblib
# ---------------------------------------------------------
model_path = "models/loan_approval_model.pkl"
model = joblib.load(model_path)  # Use joblib, not pickle

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="centered")

st.title("💰 Loan Approval Prediction App")
st.markdown("Predict whether a loan will be **Approved or Not Approved** based on applicant details.")

# ---------------------------------------------------------
# User Input Form
# ---------------------------------------------------------
st.header("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, step=100)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, step=100)
    loan_amount = st.number_input("Loan Amount (in ₹ Thousands)", min_value=0, step=1)
    loan_amount_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

# ---------------------------------------------------------
# Feature Engineering (match training features exactly)
# ---------------------------------------------------------
def preprocess_input():
    # Encode categorical features as used during training
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0

    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents_val = dependents_map[dependents]

    education_val = 1 if education == "Graduate" else 0
    self_emp_val = 1 if self_employed == "Yes" else 0

    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_val = property_map[property_area]

    credit_val = 1 if credit_history == "Good (1)" else 0

    # Match exactly the 11 features used in training
    features = np.array([
        gender_val, married_val, dependents_val, education_val, self_emp_val,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_val, property_val
    ]).reshape(1, -1)

    return features

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔮 Predict Loan Approval"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {proba*100:.2f}%)")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {proba*100:.2f}%)")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & RandomForestClassifier")
