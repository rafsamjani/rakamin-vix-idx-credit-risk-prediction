"""
CREDIT RISK PREDICTION DASHBOARD
================================
Clean Streamlit Dashboard for Loan Default Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_artifacts():
    """Load model artifacts"""
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_path, 'models')
        
        model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        artifacts = joblib.load(os.path.join(model_dir, 'preprocessing_artifacts.pkl'))
        
        with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        with open(os.path.join(model_dir, 'feature_importance.json'), 'r') as f:
            feature_importance = json.load(f)
        
        with open(os.path.join(model_dir, 'encoder_mappings.json'), 'r') as f:
            encoder_mappings = json.load(f)
        
        sample_data = pd.read_csv(os.path.join(model_dir, 'sample_data.csv'))
        
        return {
            'model': model,
            'artifacts': artifacts,
            'metadata': metadata,
            'feature_importance': feature_importance,
            'encoder_mappings': encoder_mappings,
            'sample_data': sample_data,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}

# ============================================
# HELPER FUNCTIONS
# ============================================
def get_risk_category(prob):
    if prob < 0.3:
        return "LOW RISK", "🟢"
    elif prob < 0.6:
        return "MEDIUM RISK", "🟡"
    else:
        return "HIGH RISK", "🔴"

def create_gauge(probability):
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%"},
        title={'text': "Default Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#e74c3c" if probability > 0.6 else "#f39c12" if probability > 0.3 else "#27ae60"},
            'steps': [
                {'range': [0, 30], 'color': "#d5f5e3"},
                {'range': [30, 60], 'color': "#fef9e7"},
                {'range': [60, 100], 'color': "#fadbd8"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def preprocess_input(input_data, artifacts, encoder_mappings):
    """Preprocess user input"""
    df = pd.DataFrame([input_data])
    
    feature_columns = artifacts['feature_columns']
    categorical_cols = artifacts['categorical_columns']
    numerical_cols = artifacts['numerical_columns']
    label_encoders = artifacts['label_encoders']
    scaler = artifacts['scaler']
    
    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Encode categorical
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            val = str(df[col].iloc[0])
            df[col] = le.transform([val])[0] if val in le.classes_ else -1
    
    # Scale numerical
    num_cols = [c for c in numerical_cols if c in df.columns]
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])
    
    return df[feature_columns]

# ============================================
# MAIN APP
# ============================================
def main():
    data = load_artifacts()
    
    # Sidebar
    st.sidebar.title("🏦 Credit Risk")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", ["Home", "Risk Assessment", "Analytics", "Model Info"])
    
    if not data['loaded']:
        st.error(f"❌ Failed to load model: {data.get('error', 'Unknown error')}")
        st.info("Make sure model files exist in the 'models' folder")
        return
    
    # ==================== HOME ====================
    if page == "Home":
        st.title("🏦 Credit Risk Prediction Dashboard")
        st.markdown("Predict loan default probability using machine learning")
        
        st.markdown("---")
        
        # Metrics
        metadata = data['metadata']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", metadata['best_model'])
        col2.metric("ROC-AUC", f"{metadata['best_model_metrics']['ROC-AUC']:.2%}")
        col3.metric("Training Samples", f"{metadata['data_info']['train_samples']:,}")
        col4.metric("Features", metadata['data_info']['feature_count'])
        
        st.markdown("---")
        
        # Quick charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Class Distribution")
            class_dist = metadata['class_distribution']['train']
            fig = px.pie(
                values=[class_dist.get('0', class_dist.get(0, 0)), 
                       class_dist.get('1', class_dist.get(1, 0))],
                names=['Non-Default', 'Default'],
                color_discrete_sequence=['#27ae60', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Model Comparison")
            test_metrics = metadata['all_model_metrics']['test']
            models = list(test_metrics.keys())
            roc_aucs = [test_metrics[m]['ROC-AUC'] for m in models]
            fig = px.bar(x=models, y=roc_aucs, color=roc_aucs, 
                        color_continuous_scale='Blues',
                        labels={'x': 'Model', 'y': 'ROC-AUC'})
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== RISK ASSESSMENT ====================
    elif page == "Risk Assessment":
        st.title("📋 Risk Assessment")
        
        encoder_mappings = data['encoder_mappings']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Loan Details")
            
            loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 10000, 500)
            int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.0, 0.5)
            term_months = st.selectbox("Term (months)", [36, 60])
            installment = st.number_input("Monthly Installment ($)", 50.0, 1500.0, 330.0, 10.0)
            
            grade_opts = encoder_mappings.get('grade', {}).get('classes', ['A','B','C','D','E','F','G'])
            grade = st.selectbox("Grade", grade_opts)
            
            sub_grade_opts = encoder_mappings.get('sub_grade', {}).get('classes', [f"{g}{i}" for g in grade_opts for i in range(1,6)])
            sub_grade = st.selectbox("Sub Grade", sub_grade_opts)
            
            purpose_opts = encoder_mappings.get('purpose', {}).get('classes', 
                ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
            purpose = st.selectbox("Purpose", purpose_opts)
            
            st.subheader("👤 Borrower Info")
            
            annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, 5000)
            emp_length_numeric = st.slider("Employment (years)", 0.0, 10.0, 5.0, 0.5)
            
            home_opts = encoder_mappings.get('home_ownership', {}).get('classes', ['RENT', 'OWN', 'MORTGAGE'])
            home_ownership = st.selectbox("Home Ownership", home_opts)
            
            verif_opts = encoder_mappings.get('verification_status', {}).get('classes', ['Verified', 'Not Verified'])
            verification_status = st.selectbox("Verification Status", verif_opts)
            
            st.subheader("📊 Credit History")
            
            dti = st.number_input("DTI (%)", 0.0, 50.0, 15.0, 1.0)
            fico_avg = st.number_input("FICO Score", 300, 850, 700, 5)
            delinq_2yrs = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)
            inq_last_6mths = st.number_input("Inquiries (6 mths)", 0, 10, 1)
            pub_rec = st.number_input("Public Records", 0, 10, 0)
            open_acc = st.number_input("Open Accounts", 0, 50, 10)
            total_acc = st.number_input("Total Accounts", 1, 100, 25)
            revol_bal = st.number_input("Revolving Balance ($)", 0, 200000, 15000)
            revol_util = st.number_input("Revolving Util (%)", 0.0, 100.0, 50.0)
        
        with col2:
            st.subheader("📊 Prediction Result")
            
            if st.button("🔮 Predict Risk", use_container_width=True):
                # Prepare input
                input_data = {
                    'loan_amnt': loan_amnt,
                    'int_rate': int_rate,
                    'installment': installment,
                    'term_months': term_months,
                    'grade': grade,
                    'sub_grade': sub_grade,
                    'purpose': purpose,
                    'annual_inc': annual_inc,
                    'emp_length_numeric': emp_length_numeric,
                    'home_ownership': home_ownership,
                    'verification_status': verification_status,
                    'dti': dti,
                    'fico_avg': fico_avg,
                    'delinq_2yrs': delinq_2yrs,
                    'inq_last_6mths': inq_last_6mths,
                    'open_acc': open_acc,
                    'pub_rec': pub_rec,
                    'revol_bal': revol_bal,
                    'revol_util': revol_util,
                    'total_acc': total_acc,
                    'loan_to_income': loan_amnt / (annual_inc + 1),
                    'payment_to_income': (installment * 12) / (annual_inc + 1),
                    'high_utilization': 1 if revol_util > 80 else 0,
                    'has_delinquency': 1 if delinq_2yrs > 0 else 0,
                    'high_inquiries': 1 if inq_last_6mths > 2 else 0
                }
                
                try:
                    processed = preprocess_input(input_data, data['artifacts'], encoder_mappings)
                    prob = data['model'].predict_proba(processed)[0][1]
                    
                    # Show gauge
                    st.plotly_chart(create_gauge(prob), use_container_width=True)
                    
                    # Risk category
                    risk_cat, risk_icon = get_risk_category(prob)
                    st.markdown(f"### {risk_icon} {risk_cat}")
                    
                    # Recommendation
                    st.markdown("---")
                    st.subheader("💡 Recommendation")
                    
                    if prob < 0.3:
                        st.success("✅ **APPROVE** - Low risk applicant")
                        st.write("- Standard loan terms recommended")
                        st.write("- Consider competitive interest rates")
                    elif prob < 0.6:
                        st.warning("⚠️ **REVIEW** - Moderate risk")
                        st.write("- Request additional documentation")
                        st.write("- Consider higher interest rate or collateral")
                    else:
                        st.error("❌ **DECLINE** - High risk")
                        st.write("- High probability of default")
                        st.write("- If approved, require substantial collateral")
                    
                    # Risk factors
                    st.markdown("---")
                    st.subheader("🔍 Risk Factors")
                    
                    factors = []
                    if fico_avg < 650: factors.append("⚠️ Low FICO score")
                    if dti > 30: factors.append("⚠️ High DTI ratio")
                    if revol_util > 80: factors.append("⚠️ High credit utilization")
                    if delinq_2yrs > 0: factors.append("⚠️ Past delinquencies")
                    if fico_avg >= 700: factors.append("✅ Good FICO score")
                    if dti < 20: factors.append("✅ Low DTI ratio")
                    if emp_length_numeric >= 5: factors.append("✅ Stable employment")
                    
                    for f in factors:
                        st.write(f)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.info("👈 Fill in the form and click **Predict Risk**")
    
    # ==================== ANALYTICS ====================
    elif page == "Analytics":
        st.title("📊 Portfolio Analytics")
        
        metadata = data['metadata']
        feature_importance = data['feature_importance']
        sample_data = data['sample_data']
        
        # Summary
        col1, col2, col3 = st.columns(3)
        total = metadata['data_info']['train_samples'] + metadata['data_info']['val_samples'] + metadata['data_info']['test_samples']
        col1.metric("Total Samples", f"{total:,}")
        col2.metric("Imbalance Ratio", f"{metadata['imbalance_ratio']:.1f}:1")
        col3.metric("Features", metadata['data_info']['feature_count'])
        
        st.markdown("---")
        
        # Feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Feature Importance")
            model_sel = st.selectbox("Model", list(feature_importance.keys()))
            
            fi_df = pd.DataFrame(feature_importance[model_sel]).head(10)
            fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                        color='importance', color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Model Comparison")
            test_metrics = metadata['all_model_metrics']['test']
            
            models = list(test_metrics.keys())
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            fig = go.Figure()
            for m in models:
                vals = [test_metrics[m][metric] for metric in metrics]
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=metrics + [metrics[0]], name=m, fill='toself'))
            
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions
        st.markdown("---")
        st.subheader("📋 Sample Predictions")
        
        disp = sample_data[['loan_amnt', 'int_rate', 'annual_inc', 'actual_target', 'predicted_proba']].copy()
        disp.columns = ['Loan Amt', 'Int Rate', 'Income', 'Actual', 'Pred Prob']
        disp['Actual'] = disp['Actual'].map({0: 'Paid', 1: 'Default'})
        disp['Pred Prob'] = disp['Pred Prob'].apply(lambda x: f"{x:.1%}")
        st.dataframe(disp, use_container_width=True)
    
    # ==================== MODEL INFO ====================
    elif page == "Model Info":
        st.title("🔬 Model Information")
        
        metadata = data['metadata']
        
        # Best model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Best Model")
            st.metric("Model", metadata['best_model'])
            st.write(f"**Training Date:** {metadata['training_date'][:10]}")
            st.write(f"**Random State:** {metadata['random_state']}")
        
        with col2:
            st.subheader("📊 Test Metrics")
            metrics = metadata['best_model_metrics']
            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            col_a.metric("Precision", f"{metrics['Precision']:.2%}")
            col_b.metric("Recall", f"{metrics['Recall']:.2%}")
            col_b.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")
        
        st.markdown("---")
        
        # All models comparison
        st.subheader("📋 All Models Performance")
        
        tab1, tab2 = st.tabs(["Validation", "Test"])
        
        with tab1:
            val_df = pd.DataFrame(metadata['all_model_metrics']['validation']).T
            st.dataframe(val_df.style.highlight_max(axis=0).format("{:.4f}"), use_container_width=True)
        
        with tab2:
            test_df = pd.DataFrame(metadata['all_model_metrics']['test']).T
            st.dataframe(test_df.style.highlight_max(axis=0).format("{:.4f}"), use_container_width=True)
        
        st.markdown("---")
        
        # Features
        st.subheader("🔢 Features Used")
        features = metadata['data_info']['features']
        
        cols = st.columns(3)
        n = len(features) // 3 + 1
        for i, col in enumerate(cols):
            with col:
                for f in features[i*n:(i+1)*n]:
                    st.write(f"• {f}")

if __name__ == "__main__":
    main()