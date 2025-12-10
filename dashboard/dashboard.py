import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #3498db, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample lending club data for demonstration"""
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'loan_amnt': np.random.lognormal(np.log(15000), 0.6, n_samples),
        'int_rate': np.random.normal(12, 4, n_samples),
        'installment': np.random.normal(400, 200, n_samples),
        'annual_inc': np.random.lognormal(np.log(70000), 0.5, n_samples),
        'dti': np.random.gamma(2, 7, n_samples),
        'fico_score': np.random.normal(690, 70, n_samples),
        'emp_length': np.random.beta(2, 5, n_samples) * 15,
        'loan_status': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'grade': pd.cut(np.random.normal(690, 70, n_samples), 
                        bins=[0, 580, 620, 660, 700, 740, 780, 850], 
                        labels=['G', 'F', 'E', 'D', 'C', 'B', 'A']),
        'home_ownership': np.random.choice(['MORTGAGE', 'OWN', 'RENT'], n_samples, p=[0.45, 0.15, 0.4]),
        'verification_status': np.random.choice(['Verified', 'Not Verified', 'Source Verified'], n_samples, p=[0.35, 0.5, 0.15]),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 
                                    'small_business', 'other', 'vacation', 'car', 'moving', 'medical'], 
                                   n_samples, p=[0.25, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02]),
        'addr_state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ'], n_samples, p=[0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.05, 0.04])
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic values
    df['loan_amnt'] = np.clip(df['loan_amnt'], 1000, 40000)
    df['int_rate'] = np.clip(df['int_rate'], 5, 30)
    df['fico_score'] = np.clip(df['fico_score'], 300, 850)
    df['dti'] = np.clip(df['dti'], 0, 100)
    df['annual_inc'] = np.clip(df['annual_inc'], 10000, 500000)
    df['emp_length'] = np.clip(df['emp_length'], 0, 15)
    
    # Add realistic correlations
    for i in range(len(df)):
        if df.loc[i, 'fico_score'] < 600:
            df.loc[i, 'int_rate'] = min(30, df.loc[i, 'int_rate'] + np.random.uniform(5, 15))
        elif df.loc[i, 'fico_score'] > 750:
            df.loc[i, 'int_rate'] = max(5, df.loc[i, 'int_rate'] - np.random.uniform(2, 8))
        
        if df.loc[i, 'dti'] > 20 and np.random.random() < 0.3:
            df.loc[i, 'loan_status'] = 1
    
    # Add engineered features
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['interest_cost'] = df['loan_amnt'] * (df['int_rate'] / 100)
    df['installment_to_income_ratio'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Separate features and target
    X = df.drop(['loan_status'], axis=1)
    y = df['loan_status']
    
    # Separate numerical and categorical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode categorical variables
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = X_encoded.copy()
    X_scaled[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler, numerical_features, categorical_features

def train_models(X_train, y_train):
    """Train multiple models"""
    # Calculate class weights for logistic regression
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Logistic Regression
    lr_model = LogisticRegression(
        random_state=42,
        class_weight=class_weight_dict,
        max_iter=1000,
        solver='liblinear'
    )
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

def make_predictions(model, X_test, y_test):
    """Make predictions and calculate metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return y_pred, y_pred_proba, metrics

def plot_roc_curve(y_test, y_pred_proba, model_name):
    """Plot ROC curve for a model"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='red')
    ))
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Paid', 'Predicted: Default'],
        y=['Actual: Paid', 'Actual: Default'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        width=500,
        height=400
    )
    return fig

def plot_feature_importance(model, feature_names, model_name, top_n=10):
    """Plot feature importance (for Random Forest) or coefficients (for Logistic Regression)"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f'Top {top_n} Feature Importances - {model_name}'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        title = f'Top {top_n} Feature Coefficients - {model_name}'
    else:
        return None
    
    # Create a dataframe of feature importances
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig = go.Figure(go.Bar(
        x=feature_imp['importance'],
        y=feature_imp['feature'],
        orientation='h',
        marker_color='skyblue'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Importance/Coefficient',
        yaxis_title='Feature',
        height=top_n * 50 + 200
    )
    return fig

def main():
    st.markdown('<h1 class="main-header">💳 Credit Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Advanced Credit Risk Analysis for Lending Club Data</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    st.sidebar.subheader("Data Configuration")
    data_size = st.sidebar.selectbox("Sample Size", [1000, 2500, 5000], index=2)
    
    if st.sidebar.button("Refresh Data"):
        st.session_state.data_loaded = False
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading and preparing data..."):
            df = create_sample_data()
            X_train, X_test, y_train, y_test, label_encoders, scaler, num_features, cat_features = preprocess_data(df)
            lr_model, rf_model = train_models(X_train, y_train)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.label_encoders = label_encoders
            st.session_state.scaler = scaler
            st.session_state.lr_model = lr_model
            st.session_state.rf_model = rf_model
            st.session_state.num_features = num_features
            st.session_state.cat_features = cat_features
            st.session_state.data_loaded = True
    
    # Retrieve data from session state
    df = st.session_state.df
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    label_encoders = st.session_state.label_encoders
    scaler = st.session_state.scaler
    lr_model = st.session_state.lr_model
    rf_model = st.session_state.rf_model
    num_features = st.session_state.num_features
    cat_features = st.session_state.cat_features
    
    # Sidebar model selection
    st.sidebar.subheader("Model Selection")
    model_choice = st.sidebar.radio(
        "Choose Model to Evaluate",
        ["Logistic Regression", "Random Forest", "Both"]
    )
    
    # Main content
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Loans</h3>
            <h2>{}</h2>
        </div>
        """.format(f"{len(df):,}"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Default Rate</h3>
            <h2>{:.2%}</h2>
        </div>
        """.format(df['loan_status'].mean()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Loan Amount</h3>
            <h2>${:,.0f}</h2>
        </div>
        """.format(df['loan_amnt'].mean()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg FICO Score</h3>
            <h2>{:.0f}</h2>
        </div>
        """.format(df['fico_score'].mean()), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <h2>{:.2%}</h2>
        </div>
        """.format((accuracy_score(y_test, lr_model.predict(X_test)) + accuracy_score(y_test, rf_model.predict(X_test)))/2), unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Model Performance", "🔍 Feature Analysis", "📋 Predictions", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Display basic dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Default Rate:**", f"{df['loan_status'].mean():.2%}")
            st.write("**Features:**", len(df.columns)-1)
        
        with col2:
            st.write("**Numeric Features:**", len(num_features))
            st.write("**Categorical Features:**", len(cat_features))
            st.write("**Missing Values:**", df.isnull().sum().sum())
        
        # Visualization of key distributions
        st.subheader("Key Variables Distribution")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('FICO Score Distribution', 'Loan Amount Distribution', 
                           'Interest Rate Distribution', 'DTI Distribution'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(go.Histogram(x=df['fico_score'], name='FICO Score', nbinsx=50), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['loan_amnt'], name='Loan Amount', nbinsx=50), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['int_rate'], name='Interest Rate', nbinsx=50), row=2, col=1)
        fig.add_trace(go.Histogram(x=df['dti'], name='DTI', nbinsx=50), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Default rate by grade
        st.subheader("Default Rate by Loan Grade")
        grade_defaults = df.groupby('grade')['loan_status'].mean().sort_index()
        fig_grade = px.bar(
            x=grade_defaults.index,
            y=grade_defaults.values,
            labels={'x': 'Grade', 'y': 'Default Rate'},
            color=grade_defaults.values,
            color_continuous_scale='RdYlGn_r',
            title="Default Rate by Loan Grade"
        )
        fig_grade.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_grade, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Make predictions
        lr_pred, lr_pred_proba, lr_metrics = make_predictions(lr_model, X_test, y_test)
        rf_pred, rf_pred_proba, rf_metrics = make_predictions(rf_model, X_test, y_test)
        
        # Display metrics in a table
        st.subheader("Model Metrics Comparison")
        metrics_df = pd.DataFrame({
            'Logistic Regression': lr_metrics,
            'Random Forest': rf_metrics
        })
        st.table(metrics_df.round(4))
        
        # Visualize metrics
        st.subheader("Model Metrics Visualization")
        metrics_names = list(lr_metrics.keys())
        lr_values = [lr_metrics[m] for m in metrics_names]
        rf_values = [rf_metrics[m] for m in metrics_names]
        
        fig_metrics = go.Figure(data=[
            go.Bar(name='Logistic Regression', x=metrics_names, y=lr_values, marker_color='skyblue'),
            go.Bar(name='Random Forest', x=metrics_names, y=rf_values, marker_color='lightgreen')
        ])
        fig_metrics.update_layout(
            barmode='group',
            title='Model Performance Comparison',
            yaxis_title='Score',
            xaxis_title='Metrics'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # ROC Curves
        st.subheader("ROC Curves Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_roc_curve(y_test, lr_pred_proba, "Logistic Regression"), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_roc_curve(y_test, rf_pred_proba, "Random Forest"), use_container_width=True)
        
        # Confusion Matrices
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_confusion_matrix(y_test, lr_pred, "Logistic Regression"), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_confusion_matrix(y_test, rf_pred, "Random Forest"), use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        st.subheader("Most Important Features for Default Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_feature_importance(lr_model, X_test.columns, "Logistic Regression"), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_feature_importance(rf_model, X_test.columns, "Random Forest"), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        corr_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 numeric columns
        corr_matrix = df[corr_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig_corr.update_layout(
            title="Feature Correlation Heatmap",
            height=600
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Individual Prediction</h2>', unsafe_allow_html=True)
        
        st.subheader("Loan Application Input")
        st.markdown("Enter loan application details to predict default risk:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=1000)
            int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
            fico_score = st.slider("FICO Credit Score", min_value=300, max_value=850, value=680, step=10)
            emp_length = st.slider("Employment Length (Years)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
        
        with col2:
            annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=70000, step=5000)
            dti = st.slider("DTI (Debt-to-Income Ratio)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
            installment = st.slider("Installment ($)", min_value=50.0, max_value=1500.0, value=400.0, step=10.0)
            home_ownership = st.selectbox("Home Ownership", ['MORTGAGE', 'OWN', 'RENT'], index=0)
        
        with col3:
            grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2)
            verification_status = st.selectbox("Verification Status", ['Verified', 'Not Verified', 'Source Verified'], index=0)
            purpose = st.selectbox("Loan Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 
                                                   'small_business', 'other', 'vacation', 'car', 'moving', 'medical'], index=0)
            addr_state = st.selectbox("State", ['CA', 'TX', 'NY', 'FL', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ'], index=0)
        
        # Create a dataframe for the input
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'int_rate': [int_rate],
            'installment': [installment],
            'annual_inc': [annual_inc],
            'dti': [dti],
            'fico_score': [fico_score],
            'emp_length': [emp_length],
            'grade': [grade],
            'home_ownership': [home_ownership],
            'verification_status': [verification_status],
            'purpose': [purpose],
            'addr_state': [addr_state]
        })
        
        # Add engineered features
        input_data['loan_to_income_ratio'] = input_data['loan_amnt'] / (input_data['annual_inc'] + 1)
        input_data['interest_cost'] = input_data['loan_amnt'] * (input_data['int_rate'] / 100)
        input_data['installment_to_income_ratio'] = input_data['installment'] / (input_data['annual_inc'] / 12 + 1)
        
        # Encode categorical variables using stored encoders
        for col in ['grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state']:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories by using the most frequent value
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0]
                    input_data[col] = label_encoders[col].transform([mode_val])[0]
        
        # Scale numerical features using stored scaler
        input_scaled = input_data.copy()
        input_scaled[num_features] = scaler.transform(input_data[num_features])
        
        # Make predictions
        if st.button("Predict Default Risk"):
            lr_prob = lr_model.predict_proba(input_scaled)[0, 1]
            rf_prob = rf_model.predict_proba(input_scaled)[0, 1]
            avg_prob = (lr_prob + rf_prob) / 2
            
            # Determine risk level
            if avg_prob < 0.3:
                risk_class = "risk-low"
                risk_text = "LOW RISK"
                bg_color = "#e8f5e8"
            elif avg_prob < 0.6:
                risk_class = "risk-medium"
                risk_text = "MEDIUM RISK"
                bg_color = "#fff8e1"
            else:
                risk_class = "risk-high"
                risk_text = "HIGH RISK"
                bg_color = "#ffebee"
            
            # Display prediction
            st.markdown(f'''
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid;">
                <h3 style="color: black;">Prediction Result</h3>
                <p><strong>Risk Level:</strong> <span class="{risk_class}">{risk_text}</span></p>
                <p><strong>Default Probability (LR):</strong> {lr_prob:.3f} ({lr_prob*100:.1f}%)</p>
                <p><strong>Default Probability (RF):</strong> {rf_prob:.3f} ({rf_prob*100:.1f}%)</p>
                <p><strong>Average Default Probability:</strong> {avg_prob:.3f} ({avg_prob*100:.1f}%)</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Show feature contributions if possible
            st.subheader("Risk Factors Analysis")
            factors = []
            if fico_score < 650:
                factors.append("Low FICO score (below 650)")
            if dti > 20:
                factors.append("High DTI ratio (above 20%)")
            if int_rate > 15:
                factors.append("High interest rate (above 15%)")
            if loan_amnt > annual_inc * 0.2:
                factors.append("High loan amount relative to income")
            if not factors:
                factors.append("All key risk factors are within acceptable ranges")
            
            for factor in factors:
                st.write(f"⚠️ {factor}")
    
    with tab5:
        st.markdown('<h2 class="sub-header">About This Dashboard</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Credit Risk Prediction Dashboard
        
        This dashboard provides an interactive interface for analyzing and predicting credit risk using machine learning models.
        
        #### Data Source
        - Simulated Lending Club dataset (2007-2014)
        - Contains loan attributes and default status
        - 15% default rate in the sample
        
        #### Models Implemented
        1. **Logistic Regression**: Linear model for binary classification
        2. **Random Forest**: Ensemble method with decision trees
        
        #### Key Features Analyzed
        - FICO credit score
        - Interest rate
        - Debt-to-income ratio (DTI)
        - Loan amount vs. annual income
        - Employment length
        
        #### Business Impact
        - Improved loan approval decisions
        - Better risk assessment
        - Enhanced portfolio management
        - Reduced default rates
        
        This dashboard was built as part of a comprehensive machine learning project for credit risk analysis.
        """)
        
        st.markdown("---")
        st.markdown("### Model Interpretation Guide")
        st.markdown("""
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Proportion of positive predictions that were correct
        - **Recall**: Proportion of actual positives that were identified
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC-AUC**: Area under the ROC curve, measuring discrimination ability
        """)

if __name__ == "__main__":
    main()