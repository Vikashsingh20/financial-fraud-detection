import streamlit as  st
import pandas as pd
import joblib

model=joblib.load("fraud_detection_model.pkl")
st.title(" Fraud Detection prediction app")
st.markdown("Please enter the transaction details and use the predict button.")
st.divider()

transaction_type=st.selectbox("Transaction type",["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"])
amount=st.number_input("Amount",min_value=0.0,value=1000.0)
oldbalanceOrg=st.number_input("Old balance(sender)",min_value=0.0,value=10000.0)
newbalanceOrg=st.number_input("New balance(sender)",min_value=0.0,value=9000.0)
oldbalanceDest=st.number_input("Old balance(receiver)",min_value=0.0,value=0.0) 
newbalanceDest=st.number_input("New balance(receiver)",min_value=0.0,value=0.0)  

if st.button("Predict"):
    input_data=pd.DataFrame({
        "type":[transaction_type],
        "amount":[amount],
        "oldbalanceOrg":[oldbalanceOrg],
        "newbalanceOrig":[newbalanceOrg],
        "oldbalanceDest":[oldbalanceDest],
        "newbalanceDest":[newbalanceDest]
    })



    prediction=model.predict(input_data)
    if prediction[0]==1:
        st.error("This transaction can be fraud.")
    else:
        st.success("This transaction looks like  not a fraud.")