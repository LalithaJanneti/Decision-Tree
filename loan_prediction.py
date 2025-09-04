## Loan Approval Prediction using Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib

#load Dataset
data=pd.read_csv("/home/rgukt/python/Linearregreesion/Loan_Prediction/loan_prediction.csv")
print("Dataset Shape:",data.shape)
#print(data.head())
data=data.drop(columns=['Loan_ID'])
#handling missing values
for col in data.columns:
    if data[col].dtype=="object":
        data[col]=data[col].fillna(data[col].mode()[0])
    else:
        data[col]=data[col].fillna(data[col].median())


#fill missing values with median, categorical with mode

label_enc=LabelEncoder()
for col in data.columns:
    if data[col].dtype =="object":
        data[col]=label_enc.fit_transform(data[col])

#print("\nAfter Encoding:\n",data.head())

#split features and target
X=data.drop("Loan_Status",axis=1)
y=data["Loan_Status"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTrain shape:",X_train.shape,"Test shape:",X_test.shape)

#Train Decision tree

dt=DecisionTreeClassifier(max_depth=5,criterion="gini",random_state=42)
dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)

#evalution
print("\nModel Evalution:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
print("\nconfusion matrix:\n",confusion_matrix(y_test,y_pred))


#hyperparameter tuning
param_grid={
    "max_depth":[3,5,7,9],
    "criterion":["gini","entropy"],
    "min_samples_split":[2,5,10]
}
grid=GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid,cv=5,scoring="accuracy")
grid.fit(X_train,y_train)

print("\nBest parameters from GridSearch:",grid.best_params_)
print("Best CV score:",grid.best_score_)

#save model
joblib.dump(dt,"loan_approval_model.pkl")
print("Model saved as loan_approval_model.pkl")


