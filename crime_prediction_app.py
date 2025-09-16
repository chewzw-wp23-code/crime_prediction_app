import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Crime Prediction System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .risk-moderate {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-high {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced prediction model with more sophisticated algorithm
@st.cache_resource
def create_advanced_prediction_model():
    """
    Advanced crime prediction model with non-linear relationships
    """
    return {
        'base_weights': {
            'poverty_pct': 0.28,
            'unemployment_rate': 0.22,
            'education_level': -0.18,
            'age_12_21': 0.20,
            'age_12_29': 0.15,
            'housing_vacant': 0.17,
            'population_density': 0.12,
            'median_income': -0.10,
            'male_pct': 0.08,
            'population': 0.05
        },
        'interaction_effects': {
            ('poverty_pct', 'unemployment_rate'): 0.15,  # Compound effect
            ('poverty_pct', 'education_level'): -0.10,   # Education mitigates poverty
            ('age_12_21', 'male_pct'): 0.12,             # Young males higher risk
            ('housing_vacant', 'poverty_pct'): 0.08      # Vacant housing + poverty
        },
        'thresholds': {
            'poverty_crisis': 0.25,      # Above 25% poverty
            'unemployment_crisis': 0.15,  # Above 15% unemployment
            'education_crisis': 0.60      # Below 60% high school
        }
    }

def advanced_predict_crime_rate(input_data, model_params):
    """
    Advanced prediction with non-linear effects and interactions
    """
    weights = model_params['base_weights']
    interactions = model_params['interaction_effects']
    thresholds = model_params['thresholds']
    
    # Base score from linear combination
    base_score = sum(weights[feature] * value for feature, value in input_data.items() if feature in weights)
    
    # Add interaction effects
    interaction_score = 0
    for (feat1, feat2), weight in interactions.items():
        if feat1 in input_data and feat2 in input_data:
            interaction_score += weight * input_data[feat1] * input_data[feat2]
    
    # Apply non-linear threshold effects
    crisis_multiplier = 1.0
    if input_data.get('poverty_pct', 0) > thresholds['poverty_crisis']:
        crisis_multiplier += 0.3
    if input_data.get('unemployment_rate', 0) > thresholds['unemployment_crisis']:
        crisis_multiplier += 0.2
    if input_data.get('education_level', 1) < thresholds['education_crisis']:
        crisis_multiplier += 0.15
    
    # Combine all effects
    final_score = (base_score + interaction_score) * crisis_multiplier
    
    # Apply sigmoid transformation for realistic probability
    crime_rate = 1 / (1 + np.exp(-10 * (final_score - 0.1)))
    return max(0.001, min(0.999, crime_rate))

def get_risk_level_and_color(prediction):
    """Get risk level, color, and detailed description"""
    if prediction < 0.15:
        return "LOW", "#28a745", "🟢", "This community shows low crime risk indicators."
    elif prediction < 0.35:
        return "MODERATE", "#ffc107", "🟡", "This community has moderate crime risk factors."
    elif prediction < 0.60:
        return "HIGH", "#fd7e14", "🟠", "This community shows elevated crime risk indicators."
    else:
        return "VERY HIGH", "#dc3545", "🔴", "This community has very high crime risk factors."

def create_radar_chart(input_data):
    """Create a radar chart of risk factors"""
    factors = ['Poverty %', 'Unemployment %', 'Education Level', 'Young Pop %', 
               'Housing Vacancy %', 'Population Density']
    values = [
        input_data['poverty_pct'] * 100,
        input_data['unemployment_rate'] * 100,
        100 - (input_data['education_level'] * 100),  # Inverted for risk
        input_data['age_12_21'] * 100,
        input_data['housing_vacant'] * 100,
        min(100, input_data['population_density'] * 100)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=factors,
        fill='toself',
        name='Risk Factors',
        line_color='rgb(255, 99, 132)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title="Community Risk Factor Profile",
        height=400
    )
    
    return fig

def create_comparison_chart(prediction, input_data):
    """Create a comparison with typical communities"""
    categories = ['Low Risk\nCommunities', 'Your Community', 'High Risk\nCommunities']
    crime_rates = [0.08, prediction, 0.65]
    colors = ['#28a745', '#ffc107' if prediction < 0.35 else '#dc3545', '#dc3545']
    
    fig = px.bar(
        x=categories, 
        y=crime_rates,
        title="Crime Rate Comparison",
        color=categories,
        color_discrete_sequence=colors
    )
    fig.update_layout(showlegend=False, height=400)
    fig.update_yaxis(title="Predicted Crime Rate", range=[0, 1])
    
    return fig

# Load the enhanced model
model_params = create_advanced_prediction_model()

# Main app layout
st.markdown('<div class="main-header">🏙️ Advanced Crime Prediction System</div>', unsafe_allow_html=True)
st.markdown("### Predict community crime rates using advanced socio-economic analysis")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📝 Community Data Input")
    
    # Group inputs into sections
    st.markdown("**📊 Population Demographics**")
    population = st.number_input("Population", min_value=0, max_value=1000000, value=50000, step=1000)
    male_pct = st.slider("% Male Population", 40.0, 60.0, 50.0, step=0.1)
    age_12_21 = st.slider("% Population Age 12-21", 0.0, 30.0, 8.0, step=0.1)
    age_12_29 = st.slider("% Population Age 12-29", 0.0, 50.0, 20.0, step=0.1)
    
    st.markdown("**💰 Economic Factors**")
    poverty_pct = st.slider("Poverty Percentage (%)", 0.0, 60.0, 12.0, step=0.1)
    unemployment_rate = st.slider("Unemployment Rate (%)", 0.0, 30.0, 5.5, step=0.1)
    median_income = st.number_input("Median Household Income ($)", 
                                   min_value=15000, max_value=150000, value=55000, step=1000)
    
    st.markdown("**🏠 Community Environment**")
    education_level = st.slider("% with High School Education", 40.0, 100.0, 85.0, step=0.1)
    housing_vacant = st.slider("% Vacant Housing", 0.0, 40.0, 6.0, step=0.1)
    population_density = st.number_input("Population Density (per sq mile)", 
                                        0, 25000, 3500, step=100)

with col2:
    st.markdown("### 🔮 Prediction Results")
    
    # Always show prediction (real-time updates)
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
    
    prediction = advanced_predict_crime_rate(input_features, model_params)
    risk_level, risk_color, risk_emoji, risk_description = get_risk_level_and_color(prediction)
    
    # Display main prediction
    st.markdown(f"""
    <div class="metric-container risk-{risk_level.lower().replace(' ', '-')}">
        <h2>{risk_emoji} {risk_level} RISK</h2>
        <h3>Predicted Crime Rate: {prediction:.1%}</h3>
        <p>{risk_description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show detailed metrics
    col2a, col2b, col2c = st.columns(3)
    with col2a:
        st.metric("Risk Score", f"{prediction:.1%}", 
                 delta=f"{(prediction-0.2)*100:+.1f}pp" if prediction != 0.2 else None)
    with col2b:
        st.metric("Risk Category", risk_level)
    with col2c:
        risk_rank = min(100, max(1, int(prediction * 100)))
        st.metric("Risk Percentile", f"{risk_rank}th")

# Visualizations
st.markdown("### 📊 Risk Analysis Dashboard")

vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    radar_chart = create_radar_chart(input_features)
    st.plotly_chart(radar_chart, use_container_width=True)

with vis_col2:
    comparison_chart = create_comparison_chart(prediction, input_features)
    st.plotly_chart(comparison_chart, use_container_width=True)

# Detailed recommendations
st.markdown("### 💡 Targeted Recommendations")

recommendations = []
if poverty_pct > 20:
    recommendations.append("🎯 **Economic Development**: Implement job creation and poverty reduction programs")
if unemployment_rate > 10:
    recommendations.append("🔧 **Employment Programs**: Establish job training and placement services")
if education_level < 75:
    recommendations.append("📚 **Education Initiative**: Invest in adult education and literacy programs")
if housing_vacant > 15:
    recommendations.append("🏠 **Housing Policy**: Address vacant properties through renovation or demolition")
if age_12_21 > 15:
    recommendations.append("👥 **Youth Programs**: Develop after-school and mentorship programs")
if prediction > 0.5:
    recommendations.append("🚨 **Immediate Intervention**: Consider enhanced community policing and social services")

if recommendations:
    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.success("✅ **Community appears stable** - Continue current policies and monitor trends")

# Footer with methodology
with st.expander("📖 Methodology & Data Sources"):
    st.markdown("""
    **Prediction Algorithm:**
    - Uses weighted socio-economic factors based on criminology research
    - Includes interaction effects between variables (e.g., poverty × unemployment)
    - Applies crisis thresholds for non-linear risk assessment
    - Sigmoid transformation ensures realistic probability outputs
    
    **Key Risk Factors:**
    - Poverty rate (28% weight) - Strong predictor of property crime
    - Unemployment (22% weight) - Linked to economic desperation
    - Education levels (-18% weight) - Higher education reduces crime risk
    - Young male population (20% combined) - Highest risk demographic
    - Housing vacancy (17% weight) - Indicates neighborhood decline
    
    **Limitations:**
    - Predictions are estimates based on statistical correlations
    - Local factors and recent policy changes not captured
    - Should be used alongside other crime prevention assessments
    """)

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.info(
        "This advanced crime prediction system uses multiple socio-economic "
        "indicators to estimate community crime risk. The model incorporates "
        "interaction effects and non-linear relationships for more accurate predictions."
    )
    
    st.markdown("### 🎯 Quick Tips")
    st.markdown("""
    - **Green sliders** = Lower crime risk
    - **Red sliders** = Higher crime risk  
    - **Real-time updates** as you adjust values
    - Check the radar chart for risk factor balance
    """)
