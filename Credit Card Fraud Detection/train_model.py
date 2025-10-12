# train_model.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

# -------------------------------
# 1. Load Dataset
# -------------------------------
credit_card_df = pd.read_csv('data/creditcard.csv')

# -------------------------------
# 2. Visualize Imbalance
# -------------------------------
sns.countplot(x='Class', data=credit_card_df)
plt.title("Class Distribution: Legitimate vs Fraud")
plt.xticks([0, 1], ['Legitimate', 'Fraud'])
plt.show()

# -------------------------------
# 3. Feature Scaling
# -------------------------------
time_scaler = StandardScaler()
amount_scaler = StandardScaler()

credit_card_df['Time'] = time_scaler.fit_transform(credit_card_df[['Time']])
credit_card_df['Amount'] = amount_scaler.fit_transform(credit_card_df[['Amount']])

# -------------------------------
# 4. Split Features and Target
# -------------------------------
X = credit_card_df.drop('Class', axis=1)
y = credit_card_df['Class']

# -------------------------------
# 5. Train/Test Split BEFORE SMOTE
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Handle Class Imbalance with SMOTE (train only)
# -------------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------------------
# 7. Model Training
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# -------------------------------
# 8. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# -------------------------------
# 9. Save Model, Scalers & Feature Names
# -------------------------------
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/time_scaler.pkl', 'wb') as f:
    pickle.dump(time_scaler, f)

with open('models/amount_scaler.pkl', 'wb') as f:
    pickle.dump(amount_scaler, f)

# Save feature order (important for Streamlit)
feature_names = X.columns.tolist()
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("✅ Model, scalers, and feature names saved successfully.")
