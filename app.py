import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load trained model and scaler (Ensure these files are in the same folder)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Real-Time Credit Predictor", layout="wide")

# --- ACTUAL DATA MAPPING (Based on German Data Labels) ---
# Ye mappings aapke LabelEncoder ke alphabetical order (A11, A12...) ke mutabiq hain
status_map = {"A11: < 0 DM": 0, "A12: 0 - 200 DM": 1, "A13: >= 200 DM": 2, "A14: No Account": 3}
history_map = {"A30: No credits/all paid": 0, "A31: All paid at this bank": 1, "A32: Existing paid": 2, "A33: Delay in past": 3, "A34: Critical account": 4}
purpose_map = {"A40: New Car": 0, "A41: Used Car": 1, "A42: Furniture": 2, "A43: Radio/TV": 3, "A44: Appliances": 4, "A45: Repairs": 5, "A46: Education": 6, "A48: Retraining": 7, "A49: Business": 8, "A410: Others": 9}
savings_map = {"A61: < 100 DM": 0, "A62: 100-500 DM": 1, "A63: 500-1000 DM": 2, "A64: >= 1000 DM": 3, "A65: Unknown": 4}
employment_map = {"A71: Unemployed": 0, "A72: < 1 Year": 1, "A73: 1-4 Years": 2, "A74: 4-7 Years": 3, "A75: >= 7 Years": 4}
personal_map = {"A91: Male (Divorced/Sep)": 0, "A92: Female (Div/Sep/Mar)": 1, "A93: Male (Single)": 2, "A94: Male (Mar/Wid)": 3}
debtor_map = {"A101: None": 0, "A102: Co-applicant": 1, "A103: Guarantor": 2}
property_map = {"A121: Real Estate": 0, "A122: Life Insurance": 1, "A123: Car/Other": 2, "A124: No Property": 3}
other_inst_map = {"A141: Bank": 0, "A142: Stores": 1, "A143: None": 2}
housing_map = {"A151: Rent": 0, "A152: Own": 1, "A153: For Free": 2}
job_map = {"A171: Unemployed/Non-Res": 0, "A172: Unskilled Resident": 1, "A173: Skilled Official": 2, "A174: Management/Self-Emp": 3}
tel_map = {"A191: None": 0, "A192: Yes": 1}
foreign_map = {"A201: Yes": 0, "A202: No": 1}

st.title("🏦 German Credit Risk Assessment System")
st.markdown("Is dashboard mein dikhaya gaya har prediction aapke trained **Random Forest** model se real-time aa raha hai.")

# Sidebar - User Inputs
st.sidebar.header("📥 Applicant Details")

def get_user_data():
    # Model expects 20 features in specific order
    s_val = status_map[st.sidebar.selectbox("Checking Status", list(status_map.keys()))]
    duration = st.sidebar.slider("Duration (Months)", 4, 72, 24)
    h_val = history_map[st.sidebar.selectbox("Credit History", list(history_map.keys()))]
    p_val = purpose_map[st.sidebar.selectbox("Purpose", list(purpose_map.keys()))]
    amount = st.sidebar.number_input("Amount (DM)", value=5000)
    sav_val = savings_map[st.sidebar.selectbox("Savings", list(savings_map.keys()))]
    emp_val = employment_map[st.sidebar.selectbox("Employment", list(employment_map.keys()))]
    inst_rate = st.sidebar.slider("Installment Rate", 1, 4, 2)
    per_val = personal_map[st.sidebar.selectbox("Personal Status", list(personal_map.keys()))]
    debt_val = debtor_map[st.sidebar.selectbox("Other Debtors", list(debtor_map.keys()))]
    residence = st.sidebar.slider("Residence Since", 1, 4, 2)
    prop_val = property_map[st.sidebar.selectbox("Property", list(property_map.keys()))]
    age = st.sidebar.slider("Age", 18, 75, 30)
    oth_inst = other_inst_map[st.sidebar.selectbox("Other Installments", list(other_inst_map.keys()))]
    house_val = housing_map[st.sidebar.selectbox("Housing", list(housing_map.keys()))]
    credits = st.sidebar.slider("Existing Credits", 1, 4, 1)
    job_val = job_map[st.sidebar.selectbox("Job", list(job_map.keys()))]
    liable = st.sidebar.selectbox("Liable People", [1, 2])
    tel_val = tel_map[st.sidebar.selectbox("Telephone", list(tel_map.keys()))]
    for_val = foreign_map[st.sidebar.selectbox("Foreign Worker", list(foreign_map.keys()))]

    # Data structure exactly as df.columns
    data = {
        'Status': s_val, 'Duration': duration, 'History': h_val, 'Purpose': p_val,
        'Amount': amount, 'Savings': sav_val, 'Employment': emp_val,
        'InstallmentRate': inst_rate, 'PersonalStatus': per_val,
        'OtherDebtors': debt_val, 'ResidenceSince': residence,
        'Property': prop_val, 'Age': age, 'OtherInstallments': oth_inst,
        'Housing': house_val, 'ExistingCredits': credits, 'Job': job_val,
        'LiablePeople': liable, 'Telephone': tel_val, 'ForeignWorker': for_val
    }
    return pd.DataFrame(data, index=[0])

df_input = get_user_data()

# --- PREDICTION ---
st.subheader("Results")
col1, col2 = st.columns(2)

# Important: Scale before predict
df_scaled = scaler.transform(df_input)
prediction = model.predict(df_scaled)
probability = model.predict_proba(df_scaled)[0][1]

with col1:
    st.write("### Input Data (Processed)")
    st.dataframe(df_input.T)

with col2:
    st.write("### Prediction")
    if prediction[0] == 1:
        st.error("🛑 Result: HIGH RISK")
    else:
        st.success("✅ Result: LOW RISK (SAFE)")
    
    st.metric(label="Risk Probability", value=f"{probability:.2%}")
    st.progress(probability)

st.divider()
st.info("Note: Ye prediction 'RandomForestClassifier' (SMOTE trained) ke base par hai.")