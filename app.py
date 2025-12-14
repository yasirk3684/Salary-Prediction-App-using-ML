import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Salary Prediction App by Yasir Khan", page_icon="💼", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("salary_model_final.pkl")

model = load_model()

st.markdown("""
    <h2 style='text-align:center; color:white; padding:14px; border-radius:10px;
    background: linear-gradient(90deg, #4b79a1, #283e51);'>
    💼 Salary Prediction App by Yasir
    </h2>
""", unsafe_allow_html=True)

st.caption("Predict employee salary using 5 key factors.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Navigation")
    mode = st.radio("Choose Mode", ["Single Prediction", "About App"])
    st.markdown("### Model Inputs")
    st.markdown("- Years at company")
    st.markdown("- Job rate")
    st.markdown("- Overtime hours")
    st.markdown("- Department")
    st.markdown("- Country")

if mode == "Single Prediction":
    st.subheader("👤 Predict Salary")

    years = st.number_input("Years at company", min_value=0, max_value=50, value=1, step=1)

    jobrate = st.number_input("Job Rate (1–10)", 
                              min_value=0.0, 
                              max_value=10.0, 
                              value=3.0, 
                              step=1.0)

    overtime = st.number_input("Overtime Hours", 
                               min_value=0, 
                               max_value=300, 
                               value=20, 
                               step=10)

    dept = st.selectbox("Department", [
        "Quality Control", 
        "Major Mfg Projects", 
        "Manufacturing"
    ])

    country = st.selectbox("Country", [
        "Egypt", 
        "Saudi Arabia", 
        "United Arab Emirates"
    ])

    if st.button("🔮 Predict Salary", type="primary"):
        try:
            input_df = pd.DataFrame([{
                "Years": years,
                "Job Rate": jobrate,
                "Overtime Hours": overtime,
                "Department": dept,
                "Country": country
            }])

            prediction = model.predict(input_df)[0]
            
            st.balloons()
            st.success("Prediction successful!")
            st.metric("Estimated Monthly Salary", f"₹ {prediction:,.2f}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

elif mode == "About App":
    st.subheader("ℹ️ About This Project")
    st.markdown("""
This model predicts **Monthly Salary** based on 5 important employee attributes:

- **Years at Company**
- **Job Rate**
- **Overtime Hours**
- **Department**
- **Country**

The model uses:
- OneHotEncoding for categorical variables  
- Linear Regression for prediction  
- A clean ML pipeline

Created by **Yasir Khan** as a mini ML project.
""")
