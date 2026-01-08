import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from src.model_engine import CreditModel

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# Initialize Model
engine = CreditModel()

# Train on first run if missing
if not os.path.exists('xgb_model.pkl'):
    with st.spinner("Training Advanced Risk Model..."):
        engine.train()

st.title("🏦 Explainable Credit Risk Scoring")
st.markdown("Use **AI + SHAP Values** to determine loan eligibility and understand *why*.")

# --- SIDEBAR: APPLICANT DETAILS ---
st.sidebar.header("Applicant Profile")
income = st.sidebar.number_input("Annual Income ($)", 20000, 200000, 55000)
age = st.sidebar.slider("Age", 21, 70, 30)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
married = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
house = st.sidebar.selectbox("House Ownership", ["Rent", "Own", "Mortgage"])
car = st.sidebar.selectbox("Car Ownership", ["No", "Yes"])
job_years = st.sidebar.slider("Years in Current Job", 0, 20, 2)

# Encoding Inputs
input_dict = {
    'Income': income,
    'Age': age,
    'Experience': experience,
    'Married': 1 if married == "Married" else 0,
    'House_Ownership': 0 if house=="Rent" else (1 if house=="Own" else 2),
    'Car_Ownership': 1 if car=="Yes" else 0,
    'Profession': 15, # Constant for demo
    'Current_Job_Years': job_years,
    'House_Years': 5 # Constant for demo
}
input_df = pd.DataFrame([input_dict])

# --- PREDICTION SECTION ---
if st.button("Analyze Risk"):
    pred, prob, shap_values, explainer = engine.predict_explain(input_df)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Decision")
        if pred == 1:
            st.error(f"❌ LOAN REJECTED")
            st.metric("Default Probability", f"{prob:.1%}")
        else:
            st.success(f"✅ LOAN APPROVED")
            st.metric("Default Probability", f"{prob:.1%}")

    with col2:
        st.subheader("🔍 Interpretability (Why?)")
        st.write("The chart below shows which features pushed the risk score UP (Red) or DOWN (Blue).")
        
        # SHAP Visualization
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)
        
        st.info("💡 **Feature Importance:** Long bars indicate the factors that most influenced this specific decision.")
