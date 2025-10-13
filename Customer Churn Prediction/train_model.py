import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score
import numpy as np
import pickle
telecom_df=pd.read_csv('data/Customer-Churn.csv')

telecom_df['Churn'] = telecom_df['Churn'].map({'Yes':1, 'No':0})

# 3️⃣ Map binary columns

binary_cols = ['Partner','Dependents','PaperlessBilling','PhoneService','MultipleLines',

               'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',

               'StreamingTV','StreamingMovies']

for col in binary_cols:

    telecom_df[col] = telecom_df[col].map({'Yes':1,'No':0,'No internet service':0,'No phone service':0})

# 4️⃣ One-hot encode multi-class categorical columns

telecom_df = pd.get_dummies(telecom_df, columns=['gender','InternetService','Contract','PaymentMethod'])

# 5️⃣ Handle TotalCharges

telecom_df['TotalCharges'] = telecom_df['TotalCharges'].replace(' ', np.nan)

telecom_df['TotalCharges'] = pd.to_numeric(telecom_df['TotalCharges'])

telecom_df['TotalCharges'] = telecom_df['TotalCharges'].fillna(telecom_df['TotalCharges'].median())

# 6️⃣ Scale numerical columns

scaler = StandardScaler()

num_cols = ['tenure','MonthlyCharges','TotalCharges']

telecom_df[num_cols] = scaler.fit_transform(telecom_df[num_cols])

# 7️⃣ Split data

X = telecom_df.drop(columns=['customerID','Churn'])

y = telecom_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)

# 8️⃣ Handle imbalance with SMOTE

smote = SMOTE(random_state=42)

X_res, y_res = smote.fit_resample(X_train, y_train)

print('Before SMOTE:', y_train.value_counts())

print('After SMOTE:', y_res.value_counts())

# 9️⃣ Train Random Forest Model

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_res, y_res)

# 10️⃣ Predict and Evaluate

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:,1]

print("Classification Report: \n", classification_report(y_test, y_pred))

print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))

print("ROC-AUC Score : \n", roc_auc_score(y_test, y_proba))

print("Accuracy Score : \n", accuracy_score(y_test, y_pred))

# 11️⃣ Save the model

pickle.dump(model, open('models/churn_model_rf.pkl','wb'))

# 12️⃣ Save feature columns for Streamlit deployment

feature_cols = X_train.columns.tolist()

pickle.dump(feature_cols, open('models/feature_columns.pkl','wb'))

pickle.dump(scaler, open('models/scaler.pkl','wb'))

print("Model and feature columns saved successfully!")