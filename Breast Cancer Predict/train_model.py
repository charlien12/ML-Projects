import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
import joblib
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
# 2️⃣ Select 10 meaningful features (to match your example)
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
]
X = df[selected_features]
Y = df['diagnosis']
# Split the data
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Scale Feature
scaler=StandardScaler()
X_Train_Scale=scaler.fit_transform(X_Train)
X_Test_Scale=scaler.transform(X_Test)

# Logistic Model
model=LogisticRegression()
model.fit(X_Train_Scale,Y_Train)

#Evalue the model
y_pred=model.predict(X_Test_Scale)
accuracy=accuracy_score(Y_Test,y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}\n")
print(classification_report(Y_Test,y_pred,target_names=data.target_names))

# 7️⃣ Save model and scaler
joblib.dump(model, 'models/breast_cancer_model.joblib')
joblib.dump(scaler, 'models/breast_cancer_scaler.joblib')

print("✅ Model and scaler saved successfully!")