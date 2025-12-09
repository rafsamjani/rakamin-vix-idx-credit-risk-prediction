# Dashboard Visualization Guide

## Executive Summary

This document provides comprehensive guidance for the interactive credit risk dashboard built with Streamlit. The dashboard serves as a user-friendly interface for loan officers, risk managers, and business stakeholders to assess credit risk and make informed lending decisions.

## Dashboard Overview

### Purpose and Objectives
1. **Real-time Risk Assessment**: Instant credit risk evaluation for loan applications
2. **Portfolio Analysis**: Overview of existing loan portfolio risk profile
3. **Business Intelligence**: Insights for strategic decision-making
4. **Model Explainability**: Transparent model predictions with feature importance

### Target Users
- **Loan Officers**: Daily loan application assessment
- **Risk Managers**: Portfolio risk monitoring
- **Business Analysts**: Strategic insights and reporting
- **Compliance Officers**: Model audit and fairness assessment

## Dashboard Architecture

### Technology Stack
- **Frontend**: Streamlit (Python-based web framework)
- **Backend**: Python with FastAPI (for real-time predictions)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost models
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud or AWS/Azure

### File Structure
```
dashboard/
├── app.py                 # Main Streamlit application
├── components/            # Reusable UI components
│   ├── risk_assessment.py
│   ├── portfolio_analysis.py
│   └── model_explanation.py
├── utils/                # Utility functions
│   ├── data_processing.py
│   └── model_utils.py
├── assets/               # Images, CSS, and static files
├── config.py             # Configuration settings
└── requirements.txt      # Dashboard-specific dependencies
```

## Page Navigation and Layout

### Main Navigation Structure
```python
# Streamlit sidebar navigation
st.sidebar.title("Credit Risk Dashboard")
page = st.sidebar.selectbox("Select Page", [
    "Risk Assessment",
    "Portfolio Analysis",
    "Model Insights",
    "Business Intelligence",
    "Settings"
])
```

### Page 1: Risk Assessment Tool

#### Layout Design
```python
def risk_assessment_page():
    st.title("Credit Risk Assessment")

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Application input form
        st.header("Application Details")
        # Input fields for new application

    with col2:
        # Real-time risk evaluation
        st.header("Risk Evaluation")
        # Prediction results and visualizations
```

#### Input Form Components
```python
def create_application_form():
    """
    Create comprehensive loan application form
    """

    # Personal Information Section
    st.subheader("Borrower Information")

    # Numeric inputs with validation
    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=20000,
        max_value=1000000,
        value=75000,
        step=5000
    )

    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000,
        max_value=35000,
        value=15000,
        step=1000
    )

    # Categorical selections
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Debt Consolidation", "Credit Card", "Home Improvement",
         "Major Purchase", "Small Business", "Other"]
    )

    emp_length = st.selectbox(
        "Employment Length",
        ["< 1 year", "1-2 years", "2-5 years", "5-10 years", "10+ years"]
    )

    # Credit information
    fico_score = st.slider(
        "FICO Score",
        min_value=300,
        max_value=850,
        value=700,
        step=5
    )

    dti_ratio = st.slider(
        "Debt-to-Income Ratio (%)",
        min_value=0,
        max_value=50,
        value=20,
        step=1
    )

    return {
        'annual_inc': annual_income,
        'loan_amnt': loan_amount,
        'purpose': loan_purpose,
        'emp_length': emp_length,
        'fico_score': fico_score,
        'dti': dti_ratio
    }
```

#### Risk Visualization Components
```python
def display_risk_assessment(features, prediction, probability):
    """
    Display comprehensive risk assessment results
    """

    # Risk Score Display
    risk_score = probability * 100

    # Gauge chart for risk score
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk Score"},
        delta = {'reference': 15},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Recommendation Panel
    col1, col2, col3 = st.columns(3)

    with col1:
        if risk_score < 10:
            st.success("APPROVED ✅")
            st.write("Low Risk Applicant")
        elif risk_score < 20:
            st.warning("REVIEW ⚠️")
            st.write("Moderate Risk")
        else:
            st.error("REJECTED ❌")
            st.write("High Risk Applicant")

    with col2:
        st.metric("Risk Score", f"{risk_score:.1f}%")
        st.metric("Confidence", f"{(abs(probability - 0.5) * 2) * 100:.1f}%")

    with col3:
        # Calculate recommended interest rate
        base_rate = 0.05
        risk_adjustment = risk_score / 100 * 0.20
        recommended_rate = (base_rate + risk_adjustment) * 100
        st.metric("Recommended Rate", f"{recommended_rate:.1f}%")
```

### Page 2: Portfolio Analysis

#### Portfolio Overview Dashboard
```python
def portfolio_analysis_page():
    """
    Display comprehensive portfolio risk analysis
    """
    st.title("Portfolio Risk Analysis")

    # Key Metrics Row
    st.header("Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Loans", portfolio_data['total_loans'])
        st.metric("Total Value", f"${portfolio_data['total_value']:,.0f}")

    with col2:
        st.metric("Default Rate", f"{portfolio_data['default_rate']:.2f}%")
        st.metric("Portfolio Risk", portfolio_data['risk_level'])

    with col3:
        st.metric("Average Yield", f"{portfolio_data['avg_yield']:.2f}%")
        st.metric("Loss Rate", f"{portfolio_data['loss_rate']:.2f}%")

    with col4:
        st.metric("Risk-Adjusted Return", f"{portfolio_data['risk_adj_return']:.2f}%")
        st.metric("Efficiency Score", portfolio_data['efficiency'])
```

#### Risk Distribution Visualizations
```python
def create_risk_distribution_charts(portfolio_data):
    """
    Create comprehensive risk distribution visualizations
    """

    # Risk Score Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram of risk scores
    ax1.hist(portfolio_data['risk_scores'], bins=50, alpha=0.7, color='skyblue')
    ax1.axvline(portfolio_data['mean_risk'], color='red', linestyle='--')
    ax1.set_title('Portfolio Risk Score Distribution')
    ax1.set_xlabel('Risk Score (%)')
    ax1.set_ylabel('Number of Loans')

    # Risk by loan grade
    grade_risk = portfolio_data.groupby('grade')['risk_score'].mean()
    ax2.bar(grade_risk.index, grade_risk.values, color=['green', 'lightgreen',
                                                        'yellow', 'orange', 'red'])
    ax2.set_title('Average Risk by Loan Grade')
    ax2.set_xlabel('Loan Grade')
    ax2.set_ylabel('Average Risk Score (%)')

    st.pyplot(fig)

    # Risk heat map by state and loan purpose
    risk_heatmap_data = portfolio_data.pivot_table(
        values='risk_score',
        index='addr_state',
        columns='purpose',
        aggfunc='mean'
    )

    fig_heatmap = px.imshow(
        risk_heatmap_data,
        title="Risk Heatmap by State and Loan Purpose",
        labels=dict(x="Loan Purpose", y="State", color="Risk Score")
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)
```

### Page 3: Model Insights

#### Feature Importance Display
```python
def model_insights_page():
    """
    Display model interpretability and insights
    """
    st.title("Model Insights & Interpretability")

    # Global Feature Importance
    st.header("Global Feature Importance")

    # Feature importance bar chart
    feature_importance = get_feature_importance()

    fig_importance = px.bar(
        x=feature_importance.values,
        y=feature_importance.index,
        orientation='h',
        title="Model Feature Importance",
        labels={'x': 'Importance Score', 'y': 'Features'}
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # SHAP explanations
    st.header("Prediction Explanations")

    # Local explanation for sample applications
    sample_application = select_sample_application()

    # Create SHAP force plot
    shap_explanation = create_shap_explanation(sample_application)
    st.pyplot(shap_explanation)

    # Feature impact visualization
    st.header("Feature Impact Analysis")

    # Partial dependence plots
    create_partial_dependence_plots()
```

#### Partial Dependence Plots
```python
def create_partial_dependence_plots():
    """
    Create partial dependence plots for key features
    """

    key_features = ['fico_score', 'annual_inc', 'dti', 'loan_amnt']

    for feature in key_features:
        pdp_data = calculate_partial_dependence(feature)

        fig_pdp = go.Figure()
        fig_pdp.add_trace(go.Scatter(
            x=pdp_data['feature_values'],
            y=pdp_data['average_prediction'],
            mode='lines',
            name=f'Impact of {feature}',
            line=dict(width=3)
        ))

        fig_pdp.update_layout(
            title=f'Partial Dependence: {feature}',
            xaxis_title=feature.replace('_', ' ').title(),
            yaxis_title='Predicted Default Probability'
        )

        st.plotly_chart(fig_pdp, use_container_width=True)
```

### Page 4: Business Intelligence

#### Financial Impact Dashboard
```python
def business_intelligence_page():
    """
    Display business-focused insights and metrics
    """
    st.title("Business Intelligence Dashboard")

    # Financial Performance Section
    st.header("Financial Performance Analysis")

    # Time series of portfolio performance
    financial_metrics = calculate_financial_time_series()

    fig_financial = go.Figure()

    # Add revenue line
    fig_financial.add_trace(go.Scatter(
        x=financial_metrics['date'],
        y=financial_metrics['revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='green')
    ))

    # Add losses line
    fig_financial.add_trace(go.Scatter(
        x=financial_metrics['date'],
        y=financial_metrics['losses'],
        mode='lines+markers',
        name='Losses',
        line=dict(color='red')
    ))

    # Add net profit line
    fig_financial.add_trace(go.Scatter(
        x=financial_metrics['date'],
        y=financial_metrics['net_profit'],
        mode='lines+markers',
        name='Net Profit',
        line=dict(color='blue', width=3)
    ))

    fig_financial.update_layout(
        title="Portfolio Financial Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Amount ($)"
    )

    st.plotly_chart(fig_financial, use_container_width=True)

    # ROI Analysis
    st.header("Return on Investment Analysis")

    roi_metrics = calculate_roi_metrics()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Annual ROI", f"{roi_metrics['annual_roi']:.1f}%")
        st.metric("Risk-Adjusted ROI", f"{roi_metrics['risk_adj_roi']:.1f}%")

    with col2:
        st.metric("Loan Volume Growth", f"{roi_metrics['volume_growth']:.1f}%")
        st.metric("Profit Margin", f"{roi_metrics['profit_margin']:.1f}%")

    with col3:
        st.metric("Default Recovery Rate", f"{roi_metrics['recovery_rate']:.1f}%")
        st.metric("Capital Efficiency", f"{roi_metrics['capital_efficiency']:.2f}")
```

### Page 5: Settings and Configuration

#### Model Configuration Panel
```python
def settings_page():
    """
    Dashboard settings and model configuration
    """
    st.title("Dashboard Settings")

    # Model Configuration
    st.header("Model Configuration")

    # Risk threshold adjustment
    current_threshold = st.slider(
        "Risk Threshold (%)",
        min_value=0,
        max_value=50,
        value=int(settings['risk_threshold'] * 100),
        step=1
    )

    # Interest rate parameters
    base_rate = st.number_input(
        "Base Interest Rate (%)",
        min_value=1.0,
        max_value=20.0,
        value=settings['base_rate'],
        step=0.1
    )

    risk_premium = st.number_input(
        "Risk Premium per 1% Risk",
        min_value=0.0,
        max_value=2.0,
        value=settings['risk_premium'],
        step=0.01
    )

    # Save settings button
    if st.button("Save Settings"):
        update_settings({
            'risk_threshold': current_threshold / 100,
            'base_rate': base_rate,
            'risk_premium': risk_premium
        })
        st.success("Settings saved successfully!")
```

## User Experience Features

### Interactive Elements

#### 1. Real-time Calculations
```python
def real_time_risk_calculation():
    """
    Provide instant feedback as users input data
    """

    # Use session state to track form inputs
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    # Real-time risk score update
    if st.session_state.form_data:
        current_risk = calculate_risk_score(st.session_state.form_data)

        # Display risk indicator
        if current_risk < 0.1:
            st.markdown("🟢 **Low Risk**")
        elif current_risk < 0.2:
            st.markdown("🟡 **Moderate Risk**")
        else:
            st.markdown("🔴 **High Risk**")
```

#### 2. Scenario Analysis
```python
def scenario_analysis_tool():
    """
    Allow users to compare different loan scenarios
    """

    st.header("Scenario Comparison")

    # Create multiple scenario inputs
    scenarios = {}
    for i in range(3):
        with st.expander(f"Scenario {i+1}"):
            scenario_input = create_scenario_form(i)
            scenarios[f"scenario_{i}"] = scenario_input

    # Compare scenarios
    if st.button("Compare Scenarios"):
        comparison_results = compare_scenarios(scenarios)

        # Display comparison table
        display_comparison_table(comparison_results)

        # Visual comparison
        create_scenario_comparison_chart(comparison_results)
```

### Data Validation and Error Handling

#### Input Validation
```python
def validate_application_input(form_data):
    """
    Validate loan application inputs
    """

    errors = []
    warnings = []

    # Income validation
    if form_data['annual_inc'] < 20000:
        errors.append("Annual income must be at least $20,000")
    elif form_data['annual_inc'] < 40000:
        warnings.append("Low income may increase risk")

    # Loan amount validation
    if form_data['loan_amnt'] > form_data['annual_inc'] * 0.5:
        errors.append("Loan amount cannot exceed 50% of annual income")

    # DTI validation
    if form_data['dti'] > 40:
        errors.append("DTI ratio cannot exceed 40%")
    elif form_data['dti'] > 30:
        warnings.append("High DTI ratio increases risk")

    return errors, warnings

def display_validation_messages(errors, warnings):
    """
    Display validation messages to user
    """

    if errors:
        for error in errors:
            st.error(f"❌ {error}")

    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
```

## Performance and Scalability

### Optimization Strategies

#### 1. Caching Strategy
```python
@st.cache_data
def load_model():
    """Cache model loading"""
    return joblib.load('models/best_model.pkl')

@st.cache_data
def calculate_feature_importance():
    """Cache feature importance calculation"""
    model = load_model()
    return extract_feature_importance(model)

@st.cache_data
def get_portfolio_summary():
    """Cache portfolio data calculation"""
    return calculate_portfolio_metrics()
```

#### 2. Lazy Loading
```python
def lazy_load_data():
    """
    Load data only when needed
    """

    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = load_portfolio_data()

    return st.session_state.portfolio_data
```

#### 3. Asynchronous Processing
```python
import asyncio

async def async_model_prediction(features):
    """
    Async model prediction for better performance
    """

    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(None, model.predict, features)

    return prediction

# Usage in Streamlit
if st.button("Calculate Risk"):
    with st.spinner("Calculating risk..."):
        result = asyncio.run(async_model_prediction(application_features))
```

## Security and Authentication

### User Authentication
```python
def authenticate_user():
    """
    Implement user authentication
    """

    # Simple authentication (in production, use proper auth system)
    def check_password():
        if st.session_state["username"] in st.secrets["passwords"] and st.secrets["passwords"][st.session_state["username"]] == st.session_state["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=check_password, key="username")
        st.text_input("Password", type="password", on_change=check_password, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=check_password, key="username")
        st.text_input("Password", type="password", on_change=check_password, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True
```

### Data Protection
```python
def protect_sensitive_data(df):
    """
    Remove or mask sensitive information in displayed data
    """

    # Mask personally identifiable information
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].apply(lambda x: f"***{str(x)[-4:]}")

    # Round financial figures
    financial_columns = ['annual_inc', 'loan_amnt', 'int_rate']
    for col in financial_columns:
        if col in df.columns:
            df[col] = df[col].round(2)

    return df
```

## Deployment and Maintenance

### Deployment Configuration
```python
# config.py
class Config:
    # Database settings
    DATABASE_URL = st.secrets.get("database_url")

    # Model settings
    MODEL_PATH = "models/best_model.pkl"
    MODEL_VERSION = "1.0.0"

    # Risk thresholds
    DEFAULT_RISK_THRESHOLD = 0.20
    MAX_RISK_THRESHOLD = 0.50

    # Interest rate settings
    BASE_INTEREST_RATE = 0.05
    MAX_INTEREST_RATE = 0.30

    # Dashboard settings
    REFRESH_INTERVAL = 300  # 5 minutes
    MAX_ROWS_DISPLAY = 1000
```

### Monitoring and Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_dashboard_activity(user_action, user_id=None):
    """
    Log dashboard user activities
    """

    activity_log = {
        'timestamp': datetime.now(),
        'user_id': user_id,
        'action': user_action,
        'session_id': st.session_state.get('session_id')
    }

    logger.info(f"Dashboard Activity: {activity_log}")
```

This comprehensive dashboard guide provides a foundation for building an intuitive, informative, and secure credit risk assessment tool that serves the needs of various stakeholders while maintaining high performance and security standards.