import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os


# Generate the ROC curve
def plot_roc_curve(y_test,y_pred_proba):
     # Ensure the target is binary (0 or 1) HERE TARGET COLUMN:OUTCOME
    y_test = (y_test == 1).astype(int)  # Force binary conversion

    # calc FPR AND TPR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc_value=auc(fpr,tpr)

# Calculate the AUC score
    auc_value = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

# Plot the ROC Curve
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc_value:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Random chance line
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Streamlit app
st.title("Diabetes Prediction Model")

uploaded_file=st.file_uploader("Upload your csv file",type=["csv"])

if uploaded_file is not None:
    # Load the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.write(df.head())

    target_column=st.selectbox("Select Target Column",df.columns)

    if target_column:
        # Keep only the features that were used during training
        x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y=df[target_column]

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        # Convert target column to binary (0 or 1)
        y = (y == 1).astype(int)

        # Check if model already exists
        model_file = "diabetes_model.pkl"
        # OS routines for NT or Posix depending on what system we're on
        if os.path.exists(model_file):
            st.warning(" Pretrained model found! Loading it now...")
            with open(model_file, "rb") as file:
                model = pickle.load(file)
            # MAKE PREDICTIONS EVEN IF MODEL IS PRETRAINED
            y_pred_proba = model.predict_proba(x_test)[:, 1]
        else:
            # Train a model
            model = XGBClassifier()
            model.fit(x_train, y_train)
            y_pred_proba = model.predict_proba(x_test)[:, 1]

            # Save the model
            with open(model_file, "wb") as file:
                pickle.dump(model, file)

        # Plot ROC Curve
        plot_roc_curve(y_test, y_pred_proba)

        st.success("Model Trained and ROC Curve Plotted Successfully!")

        # Download option
        with open(model_file, "rb") as file:
            btn = st.download_button(
                label="Download Trained XGBoost Model",
                data=file,
                file_name="xgboost_model.pkl",
                mime="application/octet-stream"
            )
