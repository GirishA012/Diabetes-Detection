import streamlit as st
import pandas as pd
import joblib
import os

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved models with correct paths
models = {
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "Random_Forest.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "Decision_Tree.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "Naive_Bayes.pkl")),
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "Logistic_Regression.pkl")),
    "SVM": joblib.load(os.path.join(BASE_DIR, "SVM.pkl"))
}

# Load dataset structure for input fields
csv_path = os.path.join(BASE_DIR, "diabetes-dataset.csv")
df = pd.read_csv(csv_path)
features = df.columns[:-1]  # Assuming the last column is the target variable

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Select a model and enter the required values to get a prediction.")

# User input fields
user_input = {feature: st.number_input(f"Enter {feature}", value=0.0) for feature in features}

# Model selection
selected_model = st.selectbox("Select a model", list(models.keys()))

# Prediction button
if st.button("Predict"):
    model = models[selected_model]
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.success(f"The model predicts: {result}")
