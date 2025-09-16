import streamlit as st
import pandas as pd
import numpy as np

# Simple prediction function (no external model file needed)
@st.cache_resource
def create_prediction_model():
    """
    Creates a simple crime prediction algorithm based on 
    weighted socio-economic factors
    """
    # Weights based on common criminology research
    weights = {
        'poverty_pct': 0.25,
        'unemployment_rate': 0.20,
        'education_level': -0.15,  # negative because higher education reduces crime
        'age_12_21': 0.18,
        'age_12_29': 0.12,
        'housing_vacant': 0.15,
        'population_density': 0.10,
        'median_income': -0.08,  # negative because higher income reduces crime
        'male_pct': 0.05,
        'population': 0.02
    }
    return weights

def predict_crime_rate(input_data, weights):
    """
    Calculate crime rate based on weighted factors
    """
    crime_score = 0
    for feature, value in input_data.items():
        if feature in weights:
            crime_score += weights[feature] * value
    
    # Normalize to reasonable crime rate range (0-1)
    crime_rate = max(0, min(1, crime_score + 0.1))
    return crime_rate

# Load the prediction weights
model_weights = create_prediction_model()

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
    # Prepare input data
    input_features = {
        'population': population / 100000,
        'poverty_pct': poverty_pct / 100,
        'unemployment_rate': unemployment_rate / 100,
        'median_income': median_income / 100000,
        'education_level': education_level / 100,
        'age_12_21': age_12_21 / 100,
        'age_12_29': age_12_29 / 100,
        'male_pct': male_pct / 100,
        'housing_vacant': housing_vacant / 100,
        'population_density': population_density / 10000
    }
    
    # Make prediction using our simple model
    prediction = predict_crime_rate(input_features, model_weights)
    
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
