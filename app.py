import streamlit as st
import pandas as pd
import joblib

# Load saved models
models = {
    "Random Forest": joblib.load("Random Forest.pkl"),
    "Decision Tree": joblib.load("Decision Tree.pkl"),
    "Naive Bayes": joblib.load("Naive Bayes.pkl"),
    "Logistic Regression": joblib.load("Logistic Regression.pkl"),
    "SVM": joblib.load("SVM.pkl")
}

# Load dataset structure for input fields
df = pd.read_csv(r"C:\Users\ASUS\Downloads\DD\diabetes-dataset.csv")
features = df.columns[:-1]  # Assuming the last column is the target variable

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Select a model and enter the required values to get a prediction.")

# User input fields
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Model selection
selected_model = st.selectbox("Select a model", list(models.keys()))

# Prediction button
if st.button("Predict"):
    model = models[selected_model]
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.success(f"The model predicts: {result}")
