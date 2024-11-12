import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('Akademi.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Academy Prediction")

marital_status = st.selectbox("Marital Status", options=["Married", "Single", "Divorced"])
application_mode = st.selectbox("Application Mode", options=["Online", "Offline"])
application_order = st.number_input("Application Order", min_value=1)
course = st.number_input("Course", min_value=1)
attendance = st.selectbox("Daytime/Evening Attendance", options=["Daytime", "Evening"])
previous_qualification = st.selectbox("Previous Qualification", options=["None", "High School", "Bachelor", "Master"])
previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0)
nationality = st.selectbox("Nationality", options=["National", "International"])
mother_qualification = st.selectbox("Mother's Qualification", options=["None", "High School", "Bachelor", "Master"])
father_qualification = st.selectbox("Father's Qualification", options=["None", "High School", "Bachelor", "Master"])
mother_occupation = st.selectbox("Mother's Occupation", options=["Unemployed", "Employed"])
father_occupation = st.selectbox("Father's Occupation", options=["Unemployed", "Employed"])
admission_grade = st.number_input("Admission Grade", min_value=0.0)
displaced = st.selectbox("Displaced", options=["No", "Yes"])
educational_special_needs = st.selectbox("Educational Special Needs", options=["No", "Yes"])
debtor = st.selectbox("Debtor", options=["No", "Yes"])
tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", options=["No", "Yes"])
gender = st.selectbox("Gender", options=["Male", "Female"])
scholarship_holder = st.selectbox("Scholarship Holder", options=["No", "Yes"])
age_at_enrollment = st.number_input("Age at Enrollment", min_value=0)
international = st.selectbox("International", options=["No", "Yes"])
curricular_units_1st_sem_credited = st.number_input("Curricular Units 1st Sem (Credited)", min_value=0)
curricular_units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", min_value=0)
curricular_units_1st_sem_evaluations = st.number_input("Curricular Units 1st Sem (Evaluations)", min_value=0)
curricular_units_1st_sem_approved = st.number_input("Curricular Units 1st Sem (Approved)", min_value=0)
curricular_units_1st_sem_grade = st.number_input("Curricular Units 1st Sem (Grade)", min_value=0.0)
curricular_units_1st_sem_without_evaluations = st.number_input("Curricular Units 1st Sem (Without Evaluations)", min_value=0)
curricular_units_2nd_sem_credited = st.number_input("Curricular Units 2nd Sem (Credited)", min_value=0)
curricular_units_2nd_sem_enrolled = st.number_input("Curricular Units 2nd Sem (Enrolled)", min_value=0)
curricular_units_2nd_sem_evaluations = st.number_input("Curricular Units 2nd Sem (Evaluations)", min_value=0)
curricular_units_2nd_sem_approved = st.number_input("Curricular Units 2nd Sem (Approved)", min_value=0)
curricular_units_2nd_sem_grade = st.number_input("Curricular Units 2nd Sem (Grade)", min_value=0.0)
curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular Units 2nd Sem (Without Evaluations)", min_value=0)
unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0)
inflation_rate = st.number_input("Inflation Rate", min_value=0.0)
gdp = st.number_input("GDP", min_value=0.0)

input_data = pd.DataFrame({
    'Marital_status': [marital_status],
    'Application_mode': [application_mode],
    'Application_order': [application_order],
    'Course': [course],
    'Daytime/evening_attendance': [attendance],
    'Previous_qualification': [previous_qualification],
    'Previous_qualification_(grade)': [previous_qualification_grade],
    'Nacionality': [nationality],
    'Mother\'s_qualification': [mother_qualification],
    'Father\'s_qualification': [father_qualification],
    'Mother\'s_occupation': [mother_occupation],
    'Father\'s_occupation': [father_occupation],
    'Admission_grade': [admission_grade],
    'Displaced': [1 if displaced == "Yes" else 0],
    'Educational_special_needs': [1 if educational_special_needs == "Yes" else 0],
    'Debtor': [1 if debtor == "Yes" else 0],
    'Tuition_fees_up_to_date': [1 if tuition_fees_up_to_date == "Yes" else 0],
    'Gender': [gender],
    'Scholarship_holder': [1 if scholarship_holder == "Yes" else 0],
    'Age_at_enrollment': [age_at_enrollment],
    'International': [1 if international == "Yes" else 0],
    'Curricular_units_1st_sem_(credited)': [curricular_units_1st_sem_credited],
    'Curricular_units_1st_sem_(enrolled)': [curricular_units_1st_sem_enrolled],
    'Curricular_units_1st_sem_(evaluations)': [curricular_units_1st_sem_evaluations],
    'Curricular_units_1st_sem_(approved)': [curricular_units_1st_sem_approved],
    'Curricular_units_1st_sem_(grade)': [curricular_units_1st_sem_grade],
    'Curricular_units_1st_sem_(without_evaluations)': [curricular_units_1st_sem_without_evaluations],
    'Curricular_units_2nd_sem_(credited)': [curricular_units_2nd_sem_credited],
    'Curricular_units_2nd_sem_(enrolled)': [curricular_units_2nd_sem_enrolled],
    'Curricular_units_2nd_sem_(evaluations)': [curricular_units_2nd_sem_evaluations],
    'Curricular_units_2nd_sem_(approved)': [curricular_units_2nd_sem_approved],
    'Curricular_units_2nd_sem_(grade)': [curricular_units_2nd_sem_grade],
    'Curricular_units_2nd_sem_(without_evaluations)': [curricular_units_2nd_sem_without_evaluations],
    'Unemployment_rate': [unemployment_rate],
    'Inflation_rate': [inflation_rate],
    'GDP': [gdp],
})

input_data = pd.get_dummies(input_data, drop_first=True)
input_data = input_data.reindex(columns=scaler.get_feature_names_out(), fill_value=0)

if st.button('Predict'):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    st.write(f"Predicted class: {predicted_class[0]}")

