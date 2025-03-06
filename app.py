import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

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

    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], labels=['20s', '30s', '40s', '50s', '60+'])

    df['BP_Category'] = pd.cut(df['BloodPressure'], bins=[0, 60, 80, 90, 200], labels=['Low', 'Normal', 'Pre-Hypertension', 'Hypertension'])

    df['High_SkinThickness'] = (df['SkinThickness'] > df['SkinThickness'].median()).astype(int)

    df['Pedigree_Risk'] = pd.cut(df['DiabetesPedigreeFunction'], bins=[0, 0.5, 1.0, 2.5], labels=['Low', 'Medium', 'High'])

# ---- Label Encoding ----
    label_encoder = LabelEncoder()
    categorical_features = ['BMI_Category', 'Age_Group', 'BP_Category', 'Pedigree_Risk']

    for col in categorical_features:
        df[col] = label_encoder.fit_transform(df[col])

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
