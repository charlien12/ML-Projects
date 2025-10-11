import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
MODELS_DIR = 'models'
spam_df=pd.read_csv('data/spam.csv',encoding='ISO-8859-1')
spam_df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
spam_df.rename(columns={'v1':'target','v2':'text'},inplace=True)

le=LabelEncoder()
spam_df['target']=le.fit_transform(spam_df['target'])

# Train/ Test Split
X=spam_df['text']
Y=spam_df['target']
X_train,X_test,Y_Train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2,stratify=Y)

# Vectorizer
vectorizer=TfidfVectorizer(stop_words='english',max_df=1.0,min_df=1,ngram_range=(1,2))
X_train_df=vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_df, Y_Train)
y_pred_lr = log_reg.predict(X_test_tfidf)
acc_lr = accuracy_score(Y_test, y_pred_lr)
print("\n===== Logistic Regression (Balanced) =====")
print(f"Accuracy: {acc_lr:.4f}")
print(classification_report(Y_test, y_pred_lr, target_names=le.classes_))
print('Confusion matrix:\n', confusion_matrix(Y_test, y_pred_lr))

joblib.dump(log_reg, os.path.join(MODELS_DIR, 'spam_model.joblib'))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.joblib'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))


print('Saved best model and artifacts to models/')
