import streamlit as st
import numpy as np
import joblib

st.title("Loan Approval Prediction App")

# Load trained model
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

Gender = st.radio("Gender", ["Male", "Female"])
Gender = 1 if Gender == "Male" else 0

Married = st.radio("Married", ["Yes", "No"])
Married = 1 if Married == "Yes" else 0

Dependents = st.radio("Dependents", ["0", "1", "2", "3+"])
Dependents = 3 if Dependents == "3+" else int(Dependents)

Education = st.radio("Education", ["Graduate", "Not Graduate"])
Education = 0 if Education == "Graduate" else 1

Self_Employed = st.radio("Self Employed", ["Yes", "No"])
Self_Employed = 1 if Self_Employed == "Yes" else 0

ApplicantIncome = st.number_input("Applicant Income", step=1)
CoapplicantIncome = st.number_input("Coapplicant Income", step=1)
LoanAmount = st.number_input("Loan Amount", step=1)
Loan_Amount_Term = st.number_input("Loan Amount Term", step=1)

Credit_History = st.radio("Credit History", ["Yes", "No"])
Credit_History = 1 if Credit_History == "Yes" else 0

Property_Area = st.radio("Property Area", ["Rural", "Semiurban", "Urban"])
Property_Area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[Property_Area]
import pandas as pd

input_df = pd.DataFrame([{
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area": Property_Area
}])

# Encode categorical columns (same as training)
cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
input_df[cat_cols] = encoder.transform(input_df[cat_cols])

result = model.predict(input_df)


if st.button("Predict"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
