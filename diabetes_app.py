import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Encode
le_gender = LabelEncoder()
le_smoke = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])

# Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# UI
st.title("🩺 Diabetes Prediction App")

gender = st.selectbox("Gender", le_gender.classes_)
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoke = st.selectbox("Smoking History", le_smoke.classes_)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
hba1c = st.slider("HbA1c Level", 3.0, 10.0, 5.5)
glucose = st.slider("Blood Glucose Level", 50, 300, 100)

# Input conversion
input_data = pd.DataFrame([[
    le_gender.transform([gender])[0],
    age,
    hypertension,
    heart_disease,
    le_smoke.transform([smoke])[0],
    bmi,
    hba1c,
    glucose
]], columns=X.columns)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🔴 The person is likely to have diabetes.")
    else:
        st.success("🟢 The person is unlikely to have diabetes.")
