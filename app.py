import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("Early Student Dropout Risk Predictor")
st.write("M.Tech CSE Project | Enter student details below to predict dropout risk.")

model = joblib.load("dropout_model.pkl")

col1, col2 = st.columns(2)
with col1:
    u1_app = st.number_input("1st Sem Units Approved", min_value=0, max_value=20, value=5)
    u1_grd = st.number_input("1st Sem Average Grade (0-20)", min_value=0.0, max_value=20.0, value=12.5)
    u2_app = st.number_input("2nd Sem Units Approved", min_value=0, max_value=20, value=5)
    u2_grd = st.number_input("2nd Sem Average Grade (0-20)", min_value=0.0, max_value=20.0, value=12.0)

with col2:
    tuition = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])
    scholar = st.selectbox("Scholarship Holder?", ["Yes", "No"])
    age = st.number_input("Age at Enrollment", min_value=17, max_value=60, value=19)
    adm_grd = st.number_input("Admission Grade (0-200)", min_value=0.0, max_value=200.0, value=125.0)

if st.button("Predict Dropout Risk"):
    tuition_val = 1 if tuition == "Yes" else 0
    scholar_val = 1 if scholar == "Yes" else 0

    features = np.array([[u1_app, u1_grd, u2_app, u2_grd, tuition_val, scholar_val, age, adm_grd]])
    prob = model.predict_proba(features)[0][1] * 100

    st.subheader(f"Dropout Risk Probability: {prob:.2f}%")
    if prob >= 50:
        st.error("High Risk: Immediate academic counseling recommended.")
    else:
        st.success("Low Risk: Student is on track to continue/graduate.")
