import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
fake_df=pd.read_csv('data/Fake.csv')
true_df=pd.read_csv('data/True.csv')
# Add label: 0 = fake, 1 = real
fake_df["label"] = 0
true_df["label"] = 1
news_data=pd.concat([fake_df,true_df],ignore_index=True)
# Optional: Drop missing values
news_data = news_data.dropna(subset=["title", "text"])

news_data['content'] = news_data['title'] + " " + news_data['text']

# Features and labels
X = news_data['content']
y = news_data['label']

# -----------------------------
# 2. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# -----------------------------
# 3. Text vectorization
# -----------------------------
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 4. Train classifier
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 5. Evaluate model
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 6. Save model and vectorizer
# -----------------------------
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")
