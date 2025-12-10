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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample lending club data for demonstration"""
    np.random.seed(42)
    n_samples = 8000
    
    data = {
        'loan_amnt': np.random.lognormal(np.log(15000), 0.6, n_samples),
        'int_rate': np.random.normal(12, 4, n_samples),
        'installment': np.random.normal(400, 200, n_samples),
        'annual_inc': np.random.lognormal(np.log(70000), 0.5, n_samples),
        'dti': np.random.gamma(2, 7, n_samples),
        'fico_score': np.random.normal(690, 70, n_samples),
        'emp_length': np.random.gamma(2, 2, n_samples) * 15,
        'loan_status': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # 15% default rate
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
        # Lower FICO scores tend to have higher interest rates
        if df.loc[i, 'fico_score'] < 600:
            df.loc[i, 'int_rate'] = min(30, df.loc[i, 'int_rate'] + np.random.uniform(5, 15))
        elif df.loc[i, 'fico_score'] > 750:
            df.loc[i, 'int_rate'] = max(5, df.loc[i, 'int_rate'] - np.random.uniform(2, 8))
        
        # Higher DTI tends to associate with higher default risk
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
    models = {}
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        class_weight=class_weight_dict,
        max_iter=1000,
        solver='liblinear'
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model

    # XGBoost
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model

    # Support Vector Machine
    print("Training SVM...")
    svm_model = SVC(
        kernel='rbf',
        random_state=42,
        class_weight='balanced',
        probability=True
    )
    svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model

    # Neural Network
    print("Training Neural Network...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        random_state=42,
        max_iter=300,
        early_stopping=True
    )
    nn_model.fit(X_train, y_train)
    models['Neural Network'] = nn_model

    return models

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
    """Plot feature importance (for Random Forest, XGBoost) or coefficients (for Logistic Regression)"""
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

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="🏦",
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
    .navigation-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: #3498db;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        margin: 5px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏦 Credit Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Advanced Credit Risk Analysis with 5 Machine Learning Models</p>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Go to Section:",
        ["🏠 Home", "📊 Overview", "📈 Model Performance", "🔍 Feature Analysis", "📋 Predictions", "📚 Model Comparison", "ℹ️ About"]
    )

    # Load data
    st.sidebar.subheader("Data Configuration")
    data_size = st.sidebar.selectbox("Sample Size", [5000, 8000, 10000], index=1)

    if st.sidebar.button("Refresh Data"):
        st.session_state.data_loaded = False

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if not st.session_state.data_loaded:
        with st.spinner("Loading and preparing data..."):
            df = create_sample_data()
            X_train, X_test, y_train, y_test, label_encoders, scaler, num_features, cat_features = preprocess_data(df)
            models = train_models(X_train, y_train)

            # Store in session state
            st.session_state.df = df
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.label_encoders = label_encoders
            st.session_state.scaler = scaler
            st.session_state.models = models
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
    models = st.session_state.models
    num_features = st.session_state.num_features
    cat_features = st.session_state.cat_features

    # Main content based on navigation
    if page == "🏠 Home":
        st.markdown('<h2 class="sub-header">Welcome to the Credit Risk Prediction Dashboard</h2>', unsafe_allow_html=True)

        st.markdown("""
        ### Credit Risk Prediction System

        This dashboard provides an interactive interface for analyzing and predicting credit risk using 5 different machine learning models:

        1. **Logistic Regression** - Linear model for binary classification
        2. **Random Forest** - Ensemble method with decision trees
        3. **XGBoost** - Gradient boosting algorithm
        4. **Support Vector Machine** - Effective in high-dimensional spaces
        5. **Neural Network** - Deep learning approach

        #### Business Context
        The goal is to predict loan defaults (churn) to help financial institutions make informed lending decisions, reduce risk, and optimize profitability.

        This system is particularly useful for:
        - Loan approval decisions
        - Risk assessment
        - Portfolio management
        - Regulatory compliance
        """)

        # Key metrics
        st.markdown('<h2 class="sub-header">Key Project Metrics</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Dataset Size</h3>
                <h2>{:,}</h2>
                <p>Data Points</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Default Rate</h3>
                <h2>{:.2%}</h2>
                <p>Of Total Loans</p>
            </div>
            """.format(df['loan_status'].mean()), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Models Tested</h3>
                <h2>5</h2>
                <p>Machine Learning</p>
            </div>
            """.format(), unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{}</h2>
                <p>Included</p>
            </div>
            """.format(len(df.columns)-1), unsafe_allow_html=True)

        with col5:
            st.markdown("""
            <div class="metric-card">
                <h3>Best Model</h3>
                <h2>-</h2>
                <p>Awaiting Analysis</p>
            </div>
            """, unsafe_allow_html=True)

        # Model performance preview
        st.markdown('<h2 class="sub-header">Model Performance Preview</h2>', unsafe_allow_html=True)

        # Generate predictions for all models
        model_performance = {}
        for model_name, model in models.items():
            _, _, metrics = make_predictions(model, X_test, y_test)
            model_performance[model_name] = metrics

        # Display performance metrics
        perf_df = pd.DataFrame(model_performance).T
        st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'))

    elif page == "📊 Overview":
        st.markdown('<h2 class="sub-header">Dataset Overview & Analysis</h2>', unsafe_allow_html=True)

        st.write("**Dataset Information**")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Dataset Shape:** {df.shape}")
            st.write(f"**Default Rate:** {df['loan_status'].mean():.2%}")
            st.write(f"**Features:** {len(df.columns)-1}")

        with col2:
            st.write(f"**Numeric Features:** {len(num_features)}")
            st.write(f"**Categorical Features:** {len(cat_features)}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")

        # Visualizations
        st.write("**Key Variables Distribution**")
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

    elif page == "📈 Model Performance":
        st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)

        # Calculate metrics for all models
        all_metrics = {}
        all_predictions = {}
        all_probabilities = {}

        for model_name, model in models.items():
            y_pred, y_pred_proba, metrics = make_predictions(model, X_test, y_test)
            all_metrics[model_name] = metrics
            all_predictions[model_name] = y_pred
            all_probabilities[model_name] = y_pred_proba

        # Display metrics in a table
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame(all_metrics).T
        st.table(metrics_df.round(4))

        # Visualize metrics
        st.subheader("Performance Visualization")

        fig_metrics = go.Figure()
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        for metric in metrics_list:
            values = [all_metrics[model][metric] for model in models.keys()]
            fig_metrics.add_trace(go.Bar(
                x=list(models.keys()),
                y=values,
                name=metric.title(),
                text=[f'{val:.3f}' for val in values],
                textposition='auto',
            ))

        fig_metrics.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        # ROC Curves
        st.subheader("ROC Curves Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_roc_curve(y_test, all_probabilities['Logistic Regression'], "Logistic Regression"), use_container_width=True)

        with col2:
            st.plotly_chart(plot_roc_curve(y_test, all_probabilities['Random Forest'], "Random Forest"), use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(plot_roc_curve(y_test, all_probabilities['XGBoost'], "XGBoost"), use_container_width=True)

        with col4:
            st.plotly_chart(plot_roc_curve(y_test, all_probabilities['Neural Network'], "Neural Network"), use_container_width=True)

    elif page == "🔍 Feature Analysis":
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)

        st.info("Feature importance analysis helps understand which factors most influence loan default prediction.")

        # Feature importance for each model
        col1, col2 = st.columns(2)

        with col1:
            if 'Random Forest' in models:
                rf_fig = plot_feature_importance(
                    models['Random Forest'], 
                    X_train.columns, 
                    "Random Forest", 
                    top_n=10
                )
                if rf_fig:
                    st.plotly_chart(rf_fig, use_container_width=True)

            if 'XGBoost' in models:
                xgb_fig = plot_feature_importance(
                    models['XGBoost'], 
                    X_train.columns, 
                    "XGBoost", 
                    top_n=10
                )
                if xgb_fig:
                    st.plotly_chart(xgb_fig, use_container_width=True)

        with col2:
            if 'Logistic Regression' in models:
                lr_fig = plot_feature_importance(
                    models['Logistic Regression'], 
                    X_train.columns, 
                    "Logistic Regression", 
                    top_n=10
                )
                if lr_fig:
                    st.plotly_chart(lr_fig, use_container_width=True)

            # Correlation heatmap of top features
            st.subheader("Feature Correlation Heatmap")
            top_features = X_train.columns[:10]  # Top 10 features
            corr_matrix = df[list(top_features)].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlBu_r',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            fig_corr.update_layout(
                title="Top 10 Features Correlation Heatmap",
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    elif page == "📋 Predictions":
        st.markdown('<h2 class="sub-header">Credit Risk Prediction</h2>', unsafe_allow_html=True)

        st.info("Enter loan application details below to predict default risk using all models.")

        # Input form for prediction
        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=1000)
            interest_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
            fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=680, step=10)
            emp_length = st.slider("Employment Length (Years)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)

        with col2:
            annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=70000, step=5000)
            dti = st.slider("DTI Ratio (%)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
            installment = st.slider("Installment ($)", min_value=50.0, max_value=1500.0, value=400.0, step=10.0)
            home_ownership = st.selectbox("Home Ownership", ['MORTGAGE', 'OWN', 'RENT'], index=0)

        with col3:
            grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2)
            verification_status = st.selectbox("Verification Status", ['Verified', 'Not Verified', 'Source Verified'], index=0)
            purpose = st.selectbox("Loan Purpose", [
                'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 
                'small_business', 'other', 'vacation', 'car', 'moving', 'medical'
            ], index=0)
            addr_state = st.selectbox("State", [
                'CA', 'TX', 'NY', 'FL', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ'
            ], index=0)

        # Prepare input data
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amount],
            'int_rate': [interest_rate],
            'installment': [installment],
            'annual_inc': [annual_income],
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

        # Encode categorical variables
        for col in ['grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state']:
            if col in input_data.columns:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    input_data[col] = 0  # Use first category as default

        # Scale numerical features
        input_scaled = input_data.copy()
        input_scaled[num_features] = scaler.transform(input_data[num_features])

        if st.button("Calculate Risk Prediction", type="primary"):
            # Make predictions with all models
            results = {}
            for model_name, model in models.items():
                pred_proba = model.predict_proba(input_scaled)[0, 1]
                pred_class = model.predict(input_scaled)[0]
                results[model_name] = {
                    'probability': pred_proba,
                    'prediction': pred_class,
                    'risk_level': 'High' if pred_proba > 0.6 else 'Medium' if pred_proba > 0.3 else 'Low'
                }

            # Calculate average probability
            avg_probability = np.mean([r['probability'] for r in results.values()])

            # Determine risk category
            if avg_probability < 0.3:
                risk_category = "Low Risk"
                risk_color = "#27ae60"
                bg_color = "#e8f5e8"
            elif avg_probability < 0.6:
                risk_category = "Medium Risk"
                risk_color = "#f39c12"
                bg_color = "#fff8e1"
            else:
                risk_category = "High Risk"
                risk_color = "#e74c3c"
                bg_color = "#ffebee"

            # Display prediction summary
            st.markdown(f'''
            <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color}; margin: 20px 0;">
                <h3 style="color: black;">Risk Assessment Result</h3>
                <p><strong>Average Default Probability:</strong> {avg_probability:.3f} ({avg_probability*100:.1f}%)</p>
                <p><strong>Risk Category:</strong> <span style="color: {risk_color}; font-weight: bold; font-size: 1.2em;">{risk_category}</span></p>
            </div>
            ''', unsafe_allow_html=True)

            # Individual model predictions
            st.subheader("Individual Model Predictions")

            col1, col2 = st.columns(2)
            model_names = list(results.keys())
            mid_point = len(model_names) // 2

            for i, (model_name, result) in enumerate(results.items()):
                color = "#e74c3c" if result['prediction'] == 1 else "#27ae60"
                pred_text = "DEFAULT" if result['prediction'] == 1 else "NO DEFAULT"

                if i < mid_point:
                    with col1:
                        st.markdown(f"""
                        <div style="border: 2px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                            <h4 style="margin-top: 0; color: {color};">{model_name}</h4>
                            <p><strong>Default Probability:</strong> {result['probability']:.3f} ({result['probability']*100:.1f}%)</p>
                            <p><strong>Prediction:</strong> <span style="font-weight: bold; color: {color};">{pred_text}</span></p>
                            <p><strong>Risk Level:</strong> <span style="color: {'red' if result['risk_level'] == 'High' else 'orange' if result['risk_level'] == 'Medium' else 'green'};">{result['risk_level']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with col2:
                        st.markdown(f"""
                        <div style="border: 2px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                            <h4 style="margin-top: 0; color: {color};">{model_name}</h4>
                            <p><strong>Default Probability:</strong> {result['probability']:.3f} ({result['probability']*100:.1f}%)</p>
                            <p><strong>Prediction:</strong> <span style="font-weight: bold; color: {color};">{pred_text}</span></p>
                            <p><strong>Risk Level:</strong> <span style="color: {'red' if result['risk_level'] == 'High' else 'orange' if result['risk_level'] == 'Medium' else 'green'};">{result['risk_level']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)

            # Business recommendation
            st.subheader("Business Recommendation")
            if avg_probability < 0.3:
                st.success("✅ LOW RISK - Consider approving with standard terms")
                st.write("This application exhibits characteristics of low default probability based on financial indicators.")
            elif avg_probability < 0.6:
                st.warning("⚠️ MEDIUM RISK - Consider additional verification requirements")
                st.write("Moderate default probability - additional scrutiny recommended before approval.")
            else:
                st.error("❌ HIGH RISK - Recommend rejection or significantly higher interest rate")
                st.write("High default probability - proceed with caution or decline application.")

    elif page == "📚 Model Comparison":
        st.markdown('<h2 class="sub-header">Comprehensive Model Comparison</h2>', unsafe_allow_html=True)

        # Prepare comparison data
        comparison_data = []
        for model_name, model in models.items():
            metrics = all_metrics[model_name]
            for metric, value in metrics.items():
                comparison_data.append({'Model': model_name, 'Metric': metric, 'Value': value})

        comparison_df = pd.DataFrame(comparison_data)

        # Visualization
        fig = px.bar(
            comparison_df,
            x='Model',
            y='Value',
            color='Metric',
            title='Model Performance Comparison',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Ranking by ROC-AUC
        st.subheader("Model Ranking by ROC-AUC Score")
        auc_scores = {model: all_metrics[model]['roc_auc'] for model in models.keys()}
        sorted_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranking_df = pd.DataFrame(sorted_auc, columns=['Model', 'ROC_AUC'])
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        st.table(ranking_df)

        # Model characteristics
        st.subheader("Model Characteristics")
        characteristics = pd.DataFrame({
            'Model': list(models.keys()),
            'Algorithm Type': ['Linear', 'Ensemble', 'Boosting', 'SVM', 'Neural Network'],
            'Training Speed': ['Fast', 'Medium', 'Medium', 'Slow', 'Slow'],
            'Interpretability': ['High', 'Medium', 'Medium', 'Low', 'Low'],
            'Overfitting Risk': ['Low', 'Medium', 'Medium', 'High', 'High'],
            'ROC_AUC': [all_metrics[m]['roc_auc'] for m in models.keys()]
        }).sort_values('ROC_AUC', ascending=False)

        st.table(characteristics)

        # Business recommendation based on performance
        best_model = characteristics.iloc[0]['Model']
        st.info(f"Based on ROC-AUC performance, the {best_model} model appears to be the top performer. "
                f"However, consider your specific business needs (interpretability, speed) when selecting "
                f"the final model for deployment.")

    elif page == "ℹ️ About":
        st.markdown('<h2 class="sub-header">About This Credit Risk Analysis Project</h2>', unsafe_allow_html=True)

        st.markdown("""
        ## End-to-End Credit Risk Analysis System

        This project demonstrates a complete data science workflow for credit risk prediction, following the CRISP-DM methodology.

        ### Project Overview
        This system implements 5 different machine learning algorithms to predict credit risk (loan defaults) for lending institutions.

        1. **Logistic Regression**: Linear model with high interpretability
        2. **Random Forest**: Ensemble method that combines multiple decision trees
        3. **XGBoost**: Gradient boosting algorithm for high performance
        4. **Support Vector Machine**: Effective in high-dimensional spaces
        5. **Neural Network**: Deep learning approach for complex patterns

        ### Business Value
        - **Risk Assessment**: Accurate prediction of loan defaults
        - **Decision Support**: Automated decision making based on risk scores
        - **Portfolio Management**: Better understanding of risk distribution
        - **Financial Protection**: Minimize losses from bad loans

        ### Technical Approach
        The system follows the CRISP-DM methodology for data mining projects:

        1. **Business Understanding**: Define credit risk problem and objectives
        2. **Data Understanding**: Explore and analyze lending data
        3. **Data Preparation**: Clean, transform, and engineer features
        4. **Modeling**: Train and validate ML algorithms
        5. **Evaluation**: Assess model performance and business impact
        6. **Deployment**: Implement real-time prediction system

        ### Key Features
        - Real-time risk prediction on loan applications
        - Comparative analysis of multiple algorithms
        - Feature importance analysis
        - Risk-based business recommendations
        - Model performance monitoring

        The system aims to help lending institutions make informed decisions, reduce default rates, and optimize their loan approval process.
        """)

if __name__ == "__main__":
    main()