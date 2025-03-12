import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve,roc_auc_score
import xgboost as xgb

import joblib


#load the dataset
df=pd.read_csv('diabetes.csv')

#preprocessing
x=df.drop('Outcome',axis=1)
y=df['Outcome']
#  X = INPUT FEATURES (Independent Variables)
# 👉 The X variable contains everything EXCEPT the Outcome column.
# okay so in preprocessing we are trynna train the model only on the features we want
# y = OUTPUT LABEL (Dependent Variable)
# 👉 The y variable contains only the Outcome column.
# 👉 It’s like saying:

# “Yo Model! This is the answer for each row. Learn from it!”
#train-test-split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train the XGBoost model
model=xgb.XGBClassifier()
model.fit(x_train,y_train)

#save the model
joblib.dump(model, 'diabetes_model.pkl')

#make predictions
predictions=model.predict(x_test)

#check accuracy
accuracy=accuracy_score(y_test,predictions)
print(f"Accuracy:{accuracy}")

#confusion matrix
conf_matrix=confusion_matrix(y_test,predictions)
sns.heatmap(conf_matrix,annot=True,fmt='d')
plt.show()




