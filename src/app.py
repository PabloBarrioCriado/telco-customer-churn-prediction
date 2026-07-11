import streamlit as st
import pandas as pd
import numpy as np
import joblib
from transformer import ServiceAggregator
import sklearn.utils._encode as encode
import sklearn.preprocessing._encoders as encoders

# =================================================================
# 🛠️ SAFE PATCH FOR NUMPY DESERIALIZATION
# =================================================================
original_check_unknown = encode._check_unknown

def safe_check_unknown(values, known_values, return_mask=False):
    try:
        return original_check_unknown(values, known_values, return_mask)
    except TypeError:
        valid_mask = np.in1d(values, known_values, assume_unique=True)
        diff = np.setdiff1d(values, known_values, assume_unique=True)
        if return_mask:
            return diff, valid_mask
        return diff

encode._check_unknown = safe_check_unknown
encoders._check_unknown = safe_check_unknown
# =================================================================

# 1. Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Telco Churn Predictor & Retention Engine")
st.markdown("This application predicts the likelihood of a customer leaving the telecom service and **automatically generates corporate retention tactics** for the sales team.")
st.divider()

# 2. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('../models/churn_logistic_regression.joblib')

modelo = load_model()

# 3. UI Layout: Input Form
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Client Profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Is Senior Citizen? (0 = NO, 1 = YES)", ["0", "1"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.subheader("🌐 Internet Services")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("💳 Contract and Payments")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charge ($)", min_value=15.0, max_value=120.0, value=50.0)

st.divider()

# 4. Prediction Logic and Decision Matrix
if st.button("🚀 Evaluate Churn Risk & Generate Action Plan", use_container_width=True):
    
    input_dict = {
        'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner], 'Dependents': [dependents],
        'tenure': [tenure], 'MultipleLines': [multiple_lines], 'InternetService': [internet_service],
        'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup], 'DeviceProtection': [device_protection],
        'TechSupport': [tech_support], 'StreamingTV': [streaming_tv], 'StreamingMovies': [streaming_movies],
        'Contract': [contract], 'PaperlessBilling': [paperless_billing], 'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges]
    }
    input_df = pd.DataFrame(input_dict)
    
    proba_fuga = modelo.predict_proba(input_df)[0][1]
    umbral_optimo = 0.48  
    
    st.subheader("Analysis Results")
    
    if proba_fuga >= umbral_optimo:
        st.error(f"⚠️ **HIGH CHURN RISK** ({proba_fuga:.1%})")
        st.write("The customer exceeds the critical risk threshold of **48%**. Immediate preventive action is strongly advised.")
        
        # --- 🧠 PRESCRIPTIVE RETENTION ENGINE ---
        st.markdown("### 💡 Recommended Next Best Actions (NBA)")
        st.write("Based on this customer's features, apply the following customized commercial triggers:")
        
        # Rule 1: High Risk & Unstable Contract Type
        if contract == "Month-to-month":
            st.info("🎯 **Contract Upgrade:** The customer has no binding contract. Offer a **15% discount for the next 6 months** in exchange for upgrading to a stable 1-Year contract today.")
            
        # Rule 2: High Financial Weight (Price Sensitivity)
        if monthly_charges > 80.0:
            st.warning("💸 **Price Optimization:** High billing ticket detected. Audit data consumption; if they aren't fully using Streaming TV/Movies, offer a optimized down-tier package before they churn to competitors based on price.")
            
        # Rule 3: High Risk with no technical relationship tie
        if tech_support == "No" and internet_service != "No":
            st.success("🛠️ **Value-Added Loyalty Tactic:** Provide **3 months of Premium Tech Support for free**. Historical telecom data proves that customers with active technical assistance have significantly higher retention rates.")
            
        # Edge case: High risk but parameters don't trigger specific offers
        if contract != "Month-to-month" and monthly_charges <= 80.0 and tech_support != "No":
            st.info("📞 **Quality Assurance Call:** The user has good commercial conditions but presents a high algorithmic risk. Schedule an urgent courtesy call to audit local network quality or un-reported service incidents.")

    else:
        st.success(f"✅ **LOW CHURN RISK** ({proba_fuga:.1%})")
        st.write("The customer is categorized as stable. No proactive discounts or retention offers are required. Maintain standard billing cycles.")