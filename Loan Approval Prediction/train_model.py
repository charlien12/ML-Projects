import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
Loan_prediction_df=pd.read_csv('data/Loan_prediction.csv')
Loan_prediction_df.drop(columns=['Loan_ID'],inplace=True)
# Handling missing values
Loan_prediction_df['Gender'].fillna(Loan_prediction_df['Gender'].mode()[0],inplace=True)
Loan_prediction_df['Married'].fillna(Loan_prediction_df['Married'].mode()[0],inplace=True)
Loan_prediction_df['Dependents'].fillna(Loan_prediction_df['Dependents'].mode()[0],inplace=True)
Loan_prediction_df['Self_Employed'].fillna(Loan_prediction_df['Self_Employed'].mode()[0],inplace=True)
Loan_prediction_df['LoanAmount'].fillna(Loan_prediction_df['LoanAmount'].median(),inplace=True)
Loan_prediction_df['Loan_Amount_Term'].fillna(Loan_prediction_df['Loan_Amount_Term'].mode()[0],inplace=True)
Loan_prediction_df['Credit_History'].fillna(Loan_prediction_df['Credit_History'].mode()[0],inplace=True)
# Encoding categorical variables
# ---------------------------------------------
label_encoders = LabelEncoder()
Loan_prediction_df['Gender']=label_encoders.fit_transform(Loan_prediction_df['Gender'])
Loan_prediction_df['Married']=label_encoders.fit_transform(Loan_prediction_df['Married'])
Loan_prediction_df['Dependents']=label_encoders.fit_transform(Loan_prediction_df['Dependents'])
Loan_prediction_df['Education']=label_encoders.fit_transform(Loan_prediction_df['Education'])
Loan_prediction_df['Self_Employed']=label_encoders.fit_transform(Loan_prediction_df['Self_Employed'])
Loan_prediction_df['Property_Area']=label_encoders.fit_transform(Loan_prediction_df['Property_Area'])
Loan_prediction_df['Loan_Status']=label_encoders.fit_transform(Loan_prediction_df['Loan_Status'])
# Feature and target variable
print(Loan_prediction_df.columns.tolist())
X=Loan_prediction_df.drop(columns=['Loan_Status'])
y=Loan_prediction_df['Loan_Status']
# Handling class imbalance using SMOTE
smote=SMOTE(random_state=42)
X_res,y_res=smote.fit_resample(X,y)
# Save the processed data
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
# Save the trained model
# import joblib
# joblib.dump(model,'models/loan_approval_model.pkl')
