import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from PIL import Image
import time

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📶", layout="wide", initial_sidebar_state="expanded")

# Paths
MODEL_PATH = 'models/churn_model.pkl'
EDA_DIR = 'app'

# Custom CSS for UI enhancements
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff1a1a;
        transform: scale(1.02);
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def main():
    st.title("📶 Telecom Customer Churn Prediction Pro")
    st.markdown("Predict customer churn using advanced machine learning with real-time analytics.")

    model = load_model()
    
    # Create Sidebar for Navigation
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=100)
    st.sidebar.title("Dashboard Menu")
    page = st.sidebar.radio("Navigate to", ["📊 Prediction Engine", "📈 Insights & EDA"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by: Siddharth Singh**")
    st.sidebar.info("This application uses a Logistic Regression model to predict the likelihood of a customer leaving the service.")

    if page == "📊 Prediction Engine":
        
        if model is None:
            st.error("Model not found! Please run the training script first.")
            return

        # Interactive layout using tabs
        tab1, tab2, tab3 = st.tabs(["👤 Demographics", "📝 Account Details", "🔌 Services"])
        
        with st.form("prediction_form"):
            with tab1:
                col1, col2 = st.columns(2)
                gender = col1.radio("Gender", ['Male', 'Female'], horizontal=True)
                senior_citizen = col2.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
                senior_citizen = 1 if senior_citizen == "Yes" else 0
                
                col3, col4 = st.columns(2)
                partner = col3.selectbox("Has Partner?", ['Yes', 'No'])
                dependents = col4.selectbox("Has Dependents?", ['Yes', 'No'])

            with tab2:
                col1, col2 = st.columns(2)
                tenure = col1.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                monthly_charges = col2.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=50.0)
                total_charges = tenure * monthly_charges
                st.caption(f"Calculated Total Charges: **${total_charges:.2f}**")
                
                col3, col4 = st.columns(2)
                contract = col3.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
                paperless_billing = col4.toggle("Use Paperless Billing", value=True)
                paperless_billing = 'Yes' if paperless_billing else 'No'
                
                payment_method = st.selectbox("Payment Method", [
                    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
                ])

            with tab3:
                st.markdown("#### Service Subscriptions")
                col1, col2, col3 = st.columns(3)
                phone_service = col1.checkbox("Phone Service", value=True)
                phone_service = 'Yes' if phone_service else 'No'
                
                multiple_lines_options = ['Yes', 'No'] if phone_service == 'Yes' else ['No phone service']
                multiple_lines = col2.selectbox("Multiple Lines", multiple_lines_options)
                
                internet_service = col3.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
                
                st.markdown("#### Add-ons")
                addon_options = ['Yes', 'No'] if internet_service != 'No' else ['No internet service']
                
                ac1, ac2, ac3 = st.columns(3)
                online_security = ac1.selectbox("Online Security", addon_options)
                online_backup = ac2.selectbox("Online Backup", addon_options)
                device_protection = ac3.selectbox("Device Protection", addon_options)
                
                ac4, ac5, ac6 = st.columns(3)
                tech_support = ac4.selectbox("Tech Support", addon_options)
                streaming_tv = ac5.selectbox("Streaming TV", addon_options)
                streaming_movies = ac6.selectbox("Streaming Movies", addon_options)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("🔮 Predict Churn Risk")

        if submit_button:
            with st.spinner('Analyzing customer profile...'):
                time.sleep(1) # Simulate complex processing for UI effect
                
                # Prepare input data
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'SeniorCitizen': [senior_citizen],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges]
                })

                # Prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

            st.markdown("---")
            st.subheader("Results Dashboard")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Churn Risk Probability", value=f"{probability:.1%}", delta="High Risk" if probability > 0.5 else "Low Risk", delta_color="inverse")
            
            with res_col2:
                st.markdown("### Risk Assessment:")
                if prediction == 1:
                    st.error("⚠️ ALERT: This customer is highly likely to churn. Recommended action: Offer a discount or retention plan immediately.")
                    st.progress(probability, text=f"Risk Level: {probability:.1%}")
                else:
                    st.success("✅ SAFE: This customer is likely to stay.")
                    st.progress(probability, text=f"Risk Level: {probability:.1%}")

    elif page == "📈 Insights & EDA":
        st.header("Model Metrics & Insights Dashboard")
        st.markdown("Visual explorations of the Telco Customer Churn dataset.")
        
        # Load and display images in an interactive way
        def display_image(filename):
            img_path = os.path.join(EDA_DIR, filename)
            if os.path.exists(img_path):
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
            else:
                st.warning(f"Image {filename} not found.")

        tab1, tab2, tab3 = st.tabs(["💰 Financial Patterns", "📅 Contract Insights", "🔮 Feature Importance"])
        
        with tab1:
            st.subheader("Financial Impact on Churn")
            col1, col2 = st.columns(2)
            with col1:
                display_image('churn_vs_monthly_charges.png')
            with col2:
                display_image('correlation_heatmap.png')
                
        with tab2:
            st.subheader("Demographics & Contracts")
            col1, col2 = st.columns(2)
            with col1:
                display_image('churn_distribution.png')
                display_image('churn_vs_tenure.png')
            with col2:
                display_image('churn_vs_contract.png')
                
        with tab3:
            st.subheader("What drives churn?")
            st.markdown("This chart explains which features our Machine Learning model found most critical in predicting churn.")
            display_image('feature_importance.png')

if __name__ == '__main__':
    main()
