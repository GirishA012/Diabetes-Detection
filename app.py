import streamlit as st
import pandas as pd
import os
import joblib

# Define the absolute path to the model folder
MODEL_DIR = r"C:\Users\ASUS\Downloads\DD"

# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"❌ Model directory not found: {MODEL_DIR}")

# Load saved models
try:
    models = {
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "Decision_Tree.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "Naive_Bayes.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "Logistic_Regression.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "SVM.pkl"))
    }
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    raise

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Select a model and enter the required values to get a prediction.")

# User input fields
user_input = {}
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

# **Apply Feature Engineering (Same as in Model Training)**
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Age Binning
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 50, 100], labels=["Young", "Middle", "Old"])

    # BMI Binning
    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])

    # Blood Pressure Binning
    df["BP_Category"] = pd.cut(df["BloodPressure"], bins=[0, 60, 80, 120], labels=["Low", "Normal", "High"])

    # SkinThickness Transformation
    df["High_SkinThickness"] = (df["SkinThickness"] > 23).astype(int)

    # Pedigree Function Risk
    df["Pedigree_Risk"] = (df["DiabetesPedigreeFunction"] > 0.5).astype(int)

    # Drop original columns if they were not used in training
    df = df.drop(["Age", "BMI", "BloodPressure", "SkinThickness", "DiabetesPedigreeFunction"], axis=1)

    return df

# Model selection
selected_model = st.selectbox("Select a model", list(models.keys()))

# Prediction button
if st.button("Predict"):
    model = models[selected_model]
    processed_input = preprocess_input(user_input)  # Apply transformations
    prediction = model.predict(processed_input)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.success(f"The model predicts: {result}")
