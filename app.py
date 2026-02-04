import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Patient Readmission Risk", layout="wide")

model = joblib.load("../model.pkl")

st.title("🏥 Patient 30-Day Readmission Risk Predictor")

st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 20, 90, 50)
prior = st.sidebar.slider("Prior Admissions", 0, 10, 2)
los = st.sidebar.slider("Length of Stay", 1, 20, 5)
labs = st.sidebar.slider("Lab Procedures", 1, 100, 40)
meds = st.sidebar.slider("Medications", 1, 30, 10)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
emergency = st.sidebar.selectbox("Emergency Admission", [0, 1])
home = st.sidebar.selectbox("Discharged to Home", [0, 1])
gender = st.sidebar.selectbox("Gender", [0, 1])

input_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "num_prior_admissions": prior,
    "length_of_stay": los,
    "num_lab_procedures": labs,
    "num_medications": meds,
    "has_diabetes": diabetes,
    "has_hypertension": hypertension,
    "emergency_admission": emergency,
    "discharge_to_home": home
}])

prediction = model.predict_proba(input_df)[0][1]

st.subheader(f"🔍 Readmission Risk: {prediction:.2%}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

st.subheader("Feature Impact on Prediction")

fig, ax = plt.subplots()
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=input_df.iloc[0],
    feature_names=input_df.columns.tolist()
))
st.pyplot(fig)