import streamlit as st
import os
import joblib
import pandas as pd

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(page_title='📩 SMS Spam Classifier', layout='centered')

# ----------------------- PATHS ------------------------------
MODELS_DIR = 'models'
model_path = os.path.join(MODELS_DIR, 'spam_model.joblib')
vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.joblib')
label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')

# ----------------------- TITLE ------------------------------
st.title("📩 SMS Spam Classifier")
st.write("Type a message below to check whether it is **HAM** or **SPAM**.")

# ----------------------- LOAD MODEL -------------------------
if not os.path.exists(model_path):
    st.error("⚠️ Model not found! Please run `python train_model.py` first.")
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

# Sidebar info
model_name = type(model).__name__
st.sidebar.success(f"✅ Model Loaded: {model_name}")

# ----------------------- SINGLE MESSAGE PREDICTION -----------------------
message = st.text_area("✉️ Enter your message here:", height=150, placeholder="Type your SMS message...")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # Transform and predict
        X = vectorizer.transform([message])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        label = label_encoder.inverse_transform([pred])[0]

        # Show HAM or SPAM clearly
        if label.lower() == "spam":
            st.error("🚨 This message is classified as **SPAM** ❌")
        else:
            st.success("✅ This message is classified as **HAM (Not Spam)** 💬")
