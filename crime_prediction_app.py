import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Load trained model safely
@st.cache_resource
def load_model():
    model_path = "crime_prediction_model.pkl"
    if not os.path.exists(model_path):
        st.error(
            f"❌ Model file not found: {model_path}\n\n"
            "Please make sure `crime_prediction_model.pkl` is in the same folder "
            "as this app, or update the path inside load_model()."
        )
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# App Title
st.title("🏙️ Community Crime Rate Prediction System")
st.markdown("This tool predicts **violent crime rates** for communities based on socio-economic factors.")

# Sidebar - Input Section
st.sidebar.header("📝 Input Community Data")

population = st.sidebar.number_input("Population", min_value=0, max_value=1000000, value=50000)
poverty_pct = st.sidebar.slider("Poverty Percentage (%)", 0.0, 100.0, 10.0)
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 50.0, 5.0)
median_income = st.sidebar.number_input("Median Household Income ($)", min_value=0, max_value=200000, value=45000)
education_level = st.sidebar.slider("% with High School Education", 0.0, 100.0, 80.0)
age_12_21 = st.sidebar.slider("% Population Age 12-21", 0.0, 30.0, 8.0)
age_12_29 = st.sidebar.slider("% Population Age 12-29", 0.0, 50.0, 20.0)
male_pct = st.sidebar.slider("% Male Population", 40.0, 60.0, 50.0)
housing_vacant = st.sidebar.slider("% Vacant Housing", 0.0, 50.0, 5.0)
population_density = st.sidebar.number_input("Population Density (per sq mile)", 0, 50000, 3000)

# Predict Button
if st.sidebar.button("🔮 Predict Crime Rate"):
    # Prepare input
    input_data = pd.DataFrame({
        'population': [population / 100000],
        'poverty_pct': [poverty_pct / 100],
        'unemployment_rate': [unemployment_rate / 100],
        'median_income': [median_income / 100000],
        'education_level': [education_level / 100],
        'age_12_21': [age_12_21 / 100],
        'age_12_29': [age_12_29 / 100],
        'male_pct': [male_pct / 100],
        'housing_vacant': [housing_vacant / 100],
        'population_density': [population_density / 10000]
    })
    
    # Prediction
    prediction = model.predict(input_data)[0]
    
    # Results
    st.subheader("📊 Prediction Results")
    st.metric("Predicted Violent Crime Rate", f"{prediction:.4f}")
    
    # Risk Assessment
    if prediction < 0.1:
        st.success("🟢 LOW RISK: This community has a low predicted crime rate.")
    elif prediction < 0.3:
        st.warning("🟡 MODERATE RISK: This community has a moderate predicted crime rate.")
    else:
        st.error("🔴 HIGH RISK: This community has a high predicted crime rate.")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    if poverty_pct > 20:
        st.write("• Consider economic development programs to reduce poverty")
    if unemployment_rate > 10:
        st.write("• Implement job training and employment programs")
    if education_level < 70:
        st.write("• Invest in educational infrastructure and programs")
    if housing_vacant > 15:
        st.write("• Address vacant housing through urban renewal initiatives")
