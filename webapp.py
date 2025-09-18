import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Banking Customer Churn Predictor")

# ---- Categorical Inputs ----
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Student", "Retired", "Unemployed"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
account_type = st.selectbox("Account Type", ["Savings", "Current", "Fixed Deposit", "Recurring Deposit"])

# ---- Numeric Inputs ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, step=1000, value=500000)
account_tenure = st.number_input("Account Tenure (Years)", min_value=0, max_value=50, value=5)
engagement_score = st.slider("Engagement Score", 0, 100, 50)
credit_limit = st.number_input("Credit Limit", min_value=0, step=1000, value=100000)
products_held = st.number_input("Products Held", min_value=1, max_value=10, value=2)
total_investment_value = st.number_input("Total Investment Value", min_value=0, step=1000, value=200000)
avg_monthly_balance = st.number_input("Average Monthly Balance", min_value=0, step=1000, value=50000)
online_banking_score = st.slider("Online Banking Score", 0, 10, 5)
complaints_filed = st.number_input("Complaints Filed", min_value=0, max_value=20, value=0)
overdue_count = st.number_input("Overdue Count", min_value=0, max_value=20, value=0)
feedback_score = st.slider("Feedback Score", 0, 10, 5)
dormancy_period = st.number_input("Dormancy Period (Months)", min_value=0, max_value=60, value=6)

# ---- Encode Categorical (Simple Example) ----
# ⚠️ Important: You must use the same encoding as training (LabelEncoder/OneHotEncoder).
# Here I'm mapping them manually as example:
gender_map = {"Male": 0, "Female": 1}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
occupation_map = {"Salaried": 0, "Self-Employed": 1, "Student": 2, "Retired": 3, "Unemployed": 4}
region_map = {"North": 0, "South": 1, "East": 2, "West": 3}
account_map = {"Savings": 0, "Current": 1, "Fixed Deposit": 2, "Recurring Deposit": 3}

# Convert categorical inputs to numbers
gender = gender_map[gender]
marital_status = marital_map[marital_status]
occupation = occupation_map[occupation]
region = region_map[region]
account_type = account_map[account_type]






import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
# ---- Prediction ----
if st.button("Predict"):
    features = np.array([[
        age, gender, marital_status, occupation, region, account_type,
        annual_income, account_tenure, engagement_score, credit_limit,
        products_held, total_investment_value, avg_monthly_balance,
        online_banking_score, complaints_filed, overdue_count,
        feedback_score, dormancy_period
    ]])


    new_df=pd.get_dummies(features,columns=['Gender', 'Marital_Status', 'Occupation', 'Region', 'Account_Type'],drop_first=True).replace({ 'True':0, 'False':1})

    for col in new_df.select_dtypes(include='object').columns:
        new_df[col] = LabelEncoder().fit_transform(new_df[col])
    
    
    
    
    prediction = model.predict(features)
    result = "Customer will Churn ❌" if prediction[0] == 1 else "Customer will Stay ✅"
    st.success(result)
