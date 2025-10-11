# app.py
import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# 1. Load trained model & vectorizer
# -----------------------------
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or full article and find out if it's **Fake** or **Real**.")

# Text input from user
user_input = st.text_area("Paste your news here:")

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # Transform input text using TF-IDF
        input_tfidf = vectorizer.transform([user_input])
        # Predict
        prediction = model.predict(input_tfidf)[0]
        prediction_prob = model.predict_proba(input_tfidf)[0]

        # Display results
        if prediction == 0:
            st.error("❌ The news is likely FAKE")
            st.info(f"Confidence: {prediction_prob[0]*100:.2f}%")
        else:
            st.success("✅ The news is likely REAL")
            st.info(f"Confidence: {prediction_prob[1]*100:.2f}%")

# Optional: About section
st.sidebar.title("About")
st.sidebar.info(
    """
    This Fake News Detection App uses NLP and Machine Learning (TF-IDF + Logistic Regression)
    to classify news as **Fake** or **Real**.
    """
)
