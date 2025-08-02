
import streamlit as st
import numpy as np
import joblib


st.set_page_config(page_title="Alzheimer's Prediction", layout="wide")
st.title("Alzheimer's Disease Risk Prediction App")

# Load your trained model

scaler = joblib.load("scaler.pkl")
model = joblib.load("alzheimers_model.pkl")

# Dictionary mappings
gender_map = {"Male": 0, "Female": 1}
ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
education_map = {
    "None": 0,
    "High School": 1,
    "Bachelor's": 2,
    "Higher": 3,
}

def binary_radio(label):
    return 1 if st.radio(label, ['Yes', 'No']) == 'Yes' else 0

# --- Patient Information ---
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 60, 90)
gender = gender_map[st.sidebar.selectbox("Gender", list(gender_map.keys()))]
ethnicity = ethnicity_map[st.sidebar.selectbox("Ethnicity", list(ethnicity_map.keys()))]
education = education_map[st.sidebar.selectbox("Education Level", list(education_map.keys()))]



# --- Lifestyle & Medical Inputs ---
st.subheader("Lifestyle & Medical History")
smoking = binary_radio("Do they smoke?")
alcohol = st.number_input("Alcohol Consumption (units/week) - 0 to 20", 0.0, 20.0)
diet = st.number_input("Diet Quality (4 = Poor, 10 = Excellent)", 4.0, 10.0)
family_history = binary_radio("Family History of Alzheimer's?")
cvd = binary_radio("Cardiovascular Disease?")
diabetes = binary_radio("Diabetes?")
depression = binary_radio("Depression?")
head_injury = binary_radio("History of Head Injury?")
hypertension = binary_radio("Hypertension?")
systolic = st.number_input("Systolic BP (90 to 200)", 90, 200)
diastolic = st.number_input("Diastolic BP (60 to 120)", 60, 120)
physical_activity = st.number_input("Physical Activity (hrs/week) - 0 to 10", 0.0, 10.0)
sleep_quality = st.number_input("Sleep Quality (0 = Awful, 10 = Excellent)", 0.0, 10.0)
bmi = st.number_input("BMI", 15.0, 40.0)

# --- Cholesterol ---
st.subheader("Cholesterol (mg/dL)")
chol_total = st.number_input("Total Cholesterol (150 to 300)", 150.0, 300.0)
chol_ldl = st.number_input("LDL (50 to 200)", 50.0, 200.0)
chol_hdl = st.number_input("HDL (20 to 100)", 20.0, 100.0)
chol_trig = st.number_input("Triglycerides (50 to 400)", 50.0, 400.0)

# --- Cognitive and Behavioral ---
st.subheader("Cognitive Assessments")
mmse = st.number_input("MMSE Score (0 = non-cognitive, 10 = fully cognitive)", 0.0, 30.0)
adl = st.number_input("ADL Score (0 = impaired, 10 = functional)", 0.0, 10.0)
func = st.number_input("Functional Assessment (0 = impaired, 10 = functional)", 0.0, 10.0)

st.subheader("Symptoms & Behavior")
memory = binary_radio("Memory Complaints?")
behavior = binary_radio("Behavioral Problems?")
confusion = binary_radio("Confusion?")
disorientation = binary_radio("Disorientation?")
personality = binary_radio("Personality Changes?")
task_diff = binary_radio("Difficulty Completing Tasks?")
forgetfulness = binary_radio("Forgetfulness?")

# Collect all inputs
input_data = np.array([[
    age, gender, ethnicity, education, bmi, smoking, alcohol,
    physical_activity, diet, sleep_quality, family_history, cvd,
    diabetes, depression, head_injury, hypertension, systolic, diastolic,
    chol_total, chol_ldl, chol_hdl, chol_trig, mmse, func, memory,
    behavior, adl, confusion, disorientation, personality, task_diff, forgetfulness
]])

# Scale the input
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict Alzheimer's Risk"):
    prediction = model.predict(scaled_input)[0]
    label = "🟥 Positive for Alzheimer's" if prediction == 1 else "🟩 Negative for Alzheimer's"
    st.success(f"Prediction: {label}")

    # Optional: Confidence score
    proba = model.predict_proba(scaled_input)[0]
    st.info(f"Probability of Alzheimer's: {proba[1]*100:.2f}%")
