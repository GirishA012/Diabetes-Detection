import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading saved models

# Load the trained models (Make sure these are saved using joblib)
models = {
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "SVM": joblib.load("svm_model.pkl")
}

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details and select a model to predict diabetes.")

# Input fields for user to enter values
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Model selection
selected_model = st.selectbox("Select a Model", list(models.keys()))

# Prediction function
if st.button("Predict Diabetes"):
    # Convert input to numpy array and reshape
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]).reshape(1, -1)
    
    # Get the selected model and make prediction
    model = models[selected_model]
    prediction = model.predict(input_data)
    
    # Show prediction result
    if prediction[0] == 1:
        st.error("🚨 The model predicts that the person **HAS Diabetes (1)**.")
    else:
        st.success("✅ The model predicts that the person **DOES NOT HAVE Diabetes (0)**.")

# Run the app with `streamlit run app.py`
