import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and encoder
# -----------------------------
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Loan Approval Prediction App")

# -----------------------------
# User Inputs (KEEP AS STRINGS)
# -----------------------------
Gender = st.radio("Gender", ["Male", "Female"])
Married = st.radio("Married", ["Yes", "No"])
Dependents = st.radio("Dependents", ["0", "1", "2", "3+"])
Education = st.radio("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.radio("Self Employed", ["Yes", "No"])

ApplicantIncome = st.number_input("Applicant Income", step=1)
CoapplicantIncome = st.number_input("Coapplicant Income", step=1)
LoanAmount = st.number_input("Loan Amount", step=1)
Loan_Amount_Term = st.number_input("Loan Amount Term", step=1)

Credit_History = st.radio("Credit History", ["Yes", "No"])
Property_Area = st.radio("Property Area", ["Rural", "Semiurban", "Urban"])

# -----------------------------
# Fix values BEFORE dataframe
# -----------------------------
Dependents = 3 if Dependents == "3+" else int(Dependents)
Credit_History = 1 if Credit_History == "Yes" else 0

# -----------------------------
# Helper function (IMPORTANT)
# -----------------------------
def clean_input(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.fillna(0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    # Create input dataframe
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

    # Encode categorical columns
    cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    # Clean numeric issues (prevents isnan error)
    input_df = clean_input(input_df)

    # Predict
    result = model.predict(input_df)

    # Output
    if result[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")

