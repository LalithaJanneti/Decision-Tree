import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt 

#load the trained model
model=joblib.load("loan_approval_model.pkl")

#title of the app
st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the details below to check loan approval status:")



#user input fields
gender=st.selectbox("Gender",["Male","Female"])
married=st.selectbox("Marital Status",["No","Yes"])
dependents=st.selectbox("Dependents",["0","1","2","3+"])
education = st.selectbox("Education",["Graduate","NOt Graduate"])
self_employed = st.selectbox("Self Employed",["No","Yes"])
applicant_income= st.number_input("Applicant Income",min_value=0)
coapplicant_income=st.number_input("Coapplicant Income",min_value=0)
loan_amount=st.number_input("Loan Amount",min_value=0)
loan_amount_term = st.selectbox("Loan Amount Term(in months)",[12,36,60,84,120,180,240,300,360,481])
credit_history = st.selectbox("Credit History",[0.0,1.0])
property_area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])


#Encode inputs manually(median for numerical values and mode value for categorical)

def preprocess_input():
    data={
        "Gender":1 if gender=="Male" else 0,
        "Married":1 if married=="Yes" else 0,
        "Dependents": 3 if dependents=="3+" else int(dependents),
        "Education":1 if education =="Graduate" else 0,
        "Self_Employed": 1 if self_employed=="Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome":coapplicant_income,
        "LoanAmount":loan_amount,
        "Loan_Amount_Term":loan_amount_term,
        "Credit_History":credit_history,
        "Property_Area":2 if property_area=="Urban" else (1 if property_area=="Semiurban" else 0),

    }
    return pd.DataFrame([data])

#prediction

if st.button("Predict Loan Status"):
    input_df=preprocess_input()
    prediction =model.predict(input_df)[0]

    if prediction ==1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")

   
