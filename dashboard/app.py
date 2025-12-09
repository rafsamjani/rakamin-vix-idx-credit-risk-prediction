"""
Credit Risk Prediction Dashboard
==============================

Interactive Streamlit dashboard for credit risk assessment and portfolio analysis.

Author: Rafsamjani Anugrah
Date: 2024
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure warnings
warnings.filterwarnings('ignore')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained credit risk model"""
    try:
        model_path = "models/credit_risk_model_best.pkl"
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            return model_data
        else:
            st.error("Model file not found. Please ensure the model has been trained.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample loan data for demonstration"""
    try:
        # Try to load cleaned data first
        data_paths = [
            "data/processed/loan_data_cleaned.csv",
            "../data/processed/loan_data_cleaned.csv",
            "../../data/processed/loan_data_cleaned.csv"
        ]

        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, nrows=10000)  # Load sample for performance
                return df

        # If no data found, create sample data
        st.warning("No data file found. Using sample data for demonstration.")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'loan_amnt': np.random.uniform(5000, 35000, n_samples),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'annual_inc': np.random.uniform(30000, 150000, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'fico_avg': np.random.uniform(650, 800, n_samples),
        'emp_length_numeric': np.random.uniform(0, 10, n_samples),
        'loan_status_binary': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }

    # Add categorical features
    data['grade'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    data['purpose'] = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'], n_samples)
    data['home_ownership'] = np.random.choice(['MORTGAGE', 'RENT', 'OWN'], n_samples)

    return pd.DataFrame(data)

def calculate_risk_score(features, model_data):
    """Calculate risk score using the trained model"""
    try:
        model = model_data['best_model']

        # Create feature vector (simplified for demo)
        feature_vector = pd.DataFrame([features])

        # Make prediction
        risk_probability = model.predict_proba(feature_vector)[0, 1]
        risk_score = risk_probability * 100

        return risk_score, risk_probability
    except Exception as e:
        st.error(f"Error calculating risk score: {str(e)}")
        return 50, 0.5  # Default values

def get_risk_category(risk_score):
    """Get risk category based on score"""
    if risk_score < 10:
        return "Very Low", "risk-low"
    elif risk_score < 20:
        return "Low", "risk-low"
    elif risk_score < 35:
        return "Medium", "risk-medium"
    elif risk_score < 50:
        return "High", "risk-high"
    else:
        return "Very High", "risk-high"

def recommend_action(risk_score, loan_amount, annual_income):
    """Provide recommendation based on risk assessment"""
    dti_ratio = (loan_amount / (annual_income / 12)) * 100

    if risk_score < 15:
        return "✅ **APPROVED** - Low risk applicant", "green"
    elif risk_score < 30:
        if dti_ratio < 30:
            return "⚠️ **APPROVED WITH CONDITIONS** - Consider higher interest rate", "orange"
        else:
            return "❌ **REVIEW REQUIRED** - High DTI ratio", "red"
    else:
        return "❌ **REJECTED** - High risk applicant", "red"

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 25},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 30], 'color': "yellow"},
                {'range': [30, 50], 'color': "orange"},
                {'range': [50, max_value], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    """Main dashboard application"""

    # Load model and data
    model_data = load_model()
    sample_data = load_sample_data()

    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Intelligent Credit Risk Assessment for ID/X Partners</p>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🔍 Risk Assessment", "📈 Portfolio Analysis", "🎯 Model Insights", "📋 About"]
    )

    if page == "🔍 Risk Assessment":
        risk_assessment_page(model_data, sample_data)
    elif page == "📈 Portfolio Analysis":
        portfolio_analysis_page(sample_data)
    elif page == "🎯 Model Insights":
        model_insights_page(model_data)
    elif page == "📋 About":
        about_page()

def risk_assessment_page(model_data, sample_data):
    """Risk assessment page for individual applications"""

    st.header("🔍 Individual Risk Assessment")

    if model_data is None:
        st.error("Model not available. Please train the model first.")
        return

    # Create columns for form
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Application Details")

        # Loan Information
        st.write("**💰 Loan Information**")
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            max_value=35000,
            value=15000,
            step=1000
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Debt Consolidation", "Credit Card", "Home Improvement", "Major Purchase", "Small Business", "Other"]
        )

        loan_term = st.selectbox(
            "Loan Term",
            ["36 months", "60 months"]
        )

        interest_rate = st.slider(
            "Interest Rate (%)",
            min_value=5.0,
            max_value=25.0,
            value=12.0,
            step=0.1
        )

    with col2:
        st.subheader("👤 Borrower Information")

        # Borrower Information
        st.write("**👤 Personal Information**")
        annual_income = st.number_input(
            "Annual Income ($)",
            min_value=20000,
            max_value=500000,
            value=75000,
            step=5000
        )

        employment_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1-2 years", "2-5 years", "5-10 years", "10+ years"],
            index=2
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            ["MORTGAGE", "RENT", "OWN"],
            index=0
        )

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

    # Process application when button is clicked
    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):

        with st.spinner("🔄 Processing application..."):

            # Prepare features for model
            features = {
                'loan_amnt': loan_amount,
                'int_rate': interest_rate,
                'annual_inc': annual_income,
                'dti': dti_ratio,
                'fico_avg': fico_score,
                'emp_length_numeric': {'< 1 year': 0.5, '1-2 years': 1.5, '2-5 years': 3.5,
                                       '5-10 years': 7.5, '10+ years': 10}[employment_length],
                'grade': {'Debt Consolidation': 'C', 'Credit Card': 'B', 'Home Improvement': 'C',
                         'Major Purchase': 'B', 'Small Business': 'D', 'Other': 'C'}.get(loan_purpose, 'C'),
                'purpose': loan_purpose,
                'home_ownership': home_ownership,
                'term_months': 36 if loan_term == "36 months" else 60
            }

            # Calculate risk score
            risk_score, risk_probability = calculate_risk_score(features, model_data)

            # Get risk category and recommendation
            risk_category, risk_class = get_risk_category(risk_score)
            recommendation, rec_color = recommend_action(risk_score, loan_amount, annual_income)

            # Display results
            st.markdown("---")

            # Risk Score Display
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                fig = create_gauge_chart(risk_score, f"Risk Score: {risk_score:.1f}%")
                st.plotly_chart(fig, use_container_width=True)

            # Risk Category and Recommendation
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Risk Category</h4>
                    <h2 class="{risk_class}">{risk_category}</h2>
                    <p>Risk Score: {risk_score:.1f}%</p>
                    <p>Default Probability: {risk_probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Recommendation</h4>
                    <h2 style="color: {rec_color}">{recommendation}</h2>
                    <p>Based on risk assessment and borrower profile</p>
                </div>
                """, unsafe_allow_html=True)

            # Additional Details
            with st.expander("📋 Detailed Analysis"):

                # Calculate financial ratios
                monthly_income = annual_income / 12
                monthly_payment = loan_amount * (interest_rate/100/12) * (1 + interest_rate/100/12)**(36 if loan_term == "36 months" else 60) / ((1 + interest_rate/100/12)**(36 if loan_term == "36 months" else 60) - 1)
                loan_to_income = (monthly_payment / monthly_income) * 100
                effective_dti = dti_ratio + loan_to_income

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**📊 Financial Metrics**")
                    st.write(f"• Monthly Income: ${monthly_income:,.0f}")
                    st.write(f"• Monthly Payment: ${monthly_payment:,.0f}")
                    st.write(f"• Loan-to-Income Ratio: {loan_to_income:.1f}%")
                    st.write(f"• Effective DTI: {effective_dti:.1f}%")

                with col2:
                    st.write("**🎯 Risk Factors**")
                    if fico_score < 680:
                        st.write("⚠️ Low FICO score")
                    if dti_ratio > 30:
                        st.write("⚠️ High DTI ratio")
                    if employment_length in ["< 1 year", "1-2 years"]:
                        st.write("⚠️ Limited employment history")
                    if loan_to_income > 20:
                        st.write("⚠️ High loan payment burden")

def portfolio_analysis_page(sample_data):
    """Portfolio analysis page"""

    st.header("📈 Portfolio Risk Analysis")

    # Load sample data if available
    if sample_data is not None:

        # Key Metrics
        st.subheader("📊 Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_loans = len(sample_data)
            st.metric("Total Applications", f"{total_loans:,}")

        with col2:
            default_rate = sample_data['loan_status_binary'].mean() * 100
            st.metric("Default Rate", f"{default_rate:.1f}%")

        with col3:
            avg_loan_amount = sample_data['loan_amnt'].mean()
            st.metric("Avg Loan Amount", f"${avg_loan_amount:,.0f}")

        with col4:
            avg_fico = sample_data['fico_avg'].mean() if 'fico_avg' in sample_data.columns else 700
            st.metric("Avg FICO Score", f"{avg_fico:.0f}")

        # Risk Distribution
        st.subheader("🎯 Risk Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # FICO Score Distribution
            if 'fico_avg' in sample_data.columns:
                fig_fico = px.histogram(
                    sample_data,
                    x='fico_avg',
                    nbins=20,
                    title="FICO Score Distribution",
                    color_discrete_sequence=['#1f77b4']
                )
                fig_fico.update_layout(height=300)
                st.plotly_chart(fig_fico, use_container_width=True)

        with col2:
            # Loan Amount Distribution
            fig_loan = px.histogram(
                sample_data,
                x='loan_amnt',
                nbins=20,
                title="Loan Amount Distribution",
                color_discrete_sequence=['#ff7f0e']
            )
            fig_loan.update_layout(height=300)
            st.plotly_chart(fig_loan, use_container_width=True)

        # Risk by Category
        st.subheader("📋 Risk Analysis by Category")

        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["By Loan Purpose", "By FICO Category", "By Income Level"])

        with tab1:
            if 'purpose' in sample_data.columns:
                purpose_risk = sample_data.groupby('purpose')['loan_status_binary'].mean() * 100
                fig_purpose = px.bar(
                    x=purpose_risk.index,
                    y=purpose_risk.values,
                    title="Default Rate by Loan Purpose",
                    labels={'x': 'Loan Purpose', 'y': 'Default Rate (%)'}
                )
                fig_purpose.update_layout(height=400)
                st.plotly_chart(fig_purpose, use_container_width=True)

        with tab2:
            if 'fico_avg' in sample_data.columns:
                # Create FICO categories
                sample_data['fico_category'] = pd.cut(
                    sample_data['fico_avg'],
                    bins=[0, 680, 720, 760, 850],
                    labels=['Fair', 'Good', 'Very Good', 'Excellent']
                )

                fico_risk = sample_data.groupby('fico_category')['loan_status_binary'].mean() * 100
                fig_fico_cat = px.bar(
                    x=fico_risk.index,
                    y=fico_risk.values,
                    title="Default Rate by FICO Category",
                    labels={'x': 'FICO Category', 'y': 'Default Rate (%)'}
                )
                fig_fico_cat.update_layout(height=400)
                st.plotly_chart(fig_fico_cat, use_container_width=True)

        with tab3:
            if 'annual_inc' in sample_data.columns:
                # Create income categories
                sample_data['income_category'] = pd.cut(
                    sample_data['annual_inc'],
                    bins=[0, 50000, 100000, 200000, float('inf')],
                    labels=['< $50k', '$50k-$100k', '$100k-$200k', '> $200k']
                )

                income_risk = sample_data.groupby('income_category')['loan_status_binary'].mean() * 100
                fig_income = px.bar(
                    x=income_risk.index,
                    y=income_risk.values,
                    title="Default Rate by Income Level",
                    labels={'x': 'Income Category', 'y': 'Default Rate (%)'}
                )
                fig_income.update_layout(height=400)
                st.plotly_chart(fig_income, use_container_width=True)

        # Correlation Heatmap
        st.subheader("🔍 Risk Factor Correlations")

        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = sample_data[numeric_cols].corr()

            fig_heatmap = px.imshow(
                correlation_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.warning("No data available for portfolio analysis")

def model_insights_page(model_data):
    """Model insights and performance page"""

    st.header("🎯 Model Insights")

    if model_data is None:
        st.error("Model data not available. Please train the model first.")
        return

    # Model Performance Metrics
    st.subheader("📊 Model Performance")

    best_metrics = model_data.get('best_metrics', {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.3f}")

    with col2:
        st.metric("Precision", f"{best_metrics.get('precision', 0):.3f}")

    with col3:
        st.metric("Recall", f"{best_metrics.get('recall', 0):.3f}")

    with col4:
        st.metric("F1-Score", f"{best_metrics.get('f1', 0):.3f}")

    with col5:
        st.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.3f}")

    # Feature Importance (if available)
    st.subheader("🎯 Feature Importance")

    try:
        feature_importance_path = "models/feature_importance.csv"
        if os.path.exists(feature_importance_path):
            feature_importance = pd.read_csv(feature_importance_path)

            fig_importance = px.bar(
                feature_importance.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Feature Importance",
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            fig_importance.update_layout(height=600)
            st.plotly_chart(fig_importance, use_container_width=True)

        else:
            st.info("Feature importance data not available")

    except Exception as e:
        st.warning(f"Could not load feature importance: {str(e)}")

    # Business Impact Analysis
    st.subheader("💰 Business Impact Analysis")

    business_analysis = model_data.get('business_analysis', {})

    if business_analysis:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Net Financial Impact",
                f"${business_analysis.get('net_financial_impact', 0):,.0f}"
            )

            st.metric(
                "Optimal Threshold",
                f"{business_analysis.get('optimal_threshold', 0.5):.2f}"
            )

        with col2:
            st.metric(
                "Average Loan Size",
                f"${business_analysis.get('avg_loan_amount', 0):,.0f}"
            )

            st.metric(
                "Portfolio Default Rate",
                f"{business_analysis.get('default_rate', 0):.1%}"
            )

    # Model Information
    st.subheader("📋 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Details**")
        st.write(f"• Best Model: {model_data.get('best_model_name', 'Unknown')}")
        st.write(f"• Training Date: {model_data.get('model_metadata', {}).get('training_date', 'Unknown')[:10]}")
        st.write(f"• Version: {model_data.get('model_metadata', {}).get('model_version', '1.0.0')}")

        metadata = model_data.get('model_metadata', {})
        if 'dataset_shape' in metadata:
            shape = metadata['dataset_shape']
            st.write(f"• Training Samples: {shape[0]:,}")
            st.write(f"• Features: {shape[1]}")

    with col2:
        st.write("**Target Distribution**")
        target_dist = model_data.get('model_metadata', {}).get('target_distribution', {})
        if target_dist:
            total = sum(target_dist.values())
            for label, count in target_dist.items():
                status = "Fully Paid" if label == 0 else "Charged Off"
                percentage = (count / total) * 100
                st.write(f"• {status}: {count:,} ({percentage:.1f}%)")

def about_page():
    """About page"""

    st.header("📋 About This Dashboard")

    st.markdown("""
    ## Credit Risk Prediction System

    This interactive dashboard provides intelligent credit risk assessment for loan applications,
    helping lenders make informed decisions while minimizing financial losses.

    ### 🎯 Key Features

    - **Individual Risk Assessment**: Real-time evaluation of loan applications
    - **Portfolio Analysis**: Comprehensive insights into loan portfolio risk
    - **Model Insights**: Understanding of machine learning model decisions
    - **Business Impact**: Financial impact analysis and optimization

    ### 🤖 Technology Stack

    - **Machine Learning**: XGBoost, Random Forest, Logistic Regression
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    - **Visualization**: Plotly, Streamlit, Matplotlib
    - **Deployment**: Streamlit Cloud

    ### 👨‍💻 Author

    **Rafsamjani Anugrah**
    Data Scientist | Credit Risk Analyst

    *Developed for the Rakamin VIX Program in partnership with ID/X Partners*

    ### 📊 Project Information

    - **Company**: ID/X Partners
    - **Program**: Rakamin Academy Virtual Internship Experience (VIX)
    - **Dataset**: Lending Club Loan Data (2007-2014)
    - **Objective**: Predict loan default probability and minimize financial risk

    ### 🔧 How to Use

    1. **Risk Assessment**: Enter loan application details to get instant risk evaluation
    2. **Portfolio Analysis**: Analyze overall portfolio risk distribution
    3. **Model Insights**: Understand how the model makes predictions
    4. **Business Metrics**: View financial impact and key performance indicators

    ### 📈 Model Performance

    Our credit risk prediction model achieves:
    - **ROC-AUC**: 0.85+
    - **Accuracy**: 0.80+
    - **Precision**: 0.75+
    - **Recall**: 0.70+

    ### 🚀 Future Enhancements

    - Real-time API integration
    - Advanced customer segmentation
    - Automated decision workflows
    - Mobile application support

    ---

    *Last Updated: December 2024*
    """)

if __name__ == "__main__":
    main()