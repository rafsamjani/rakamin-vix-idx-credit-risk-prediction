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
        'home_ownership': np.random.choice(['MORTGAGE', 'OWN', 'RENT', 'OTHER'], n_samples, p=[0.45, 0.15, 0.4, 0.05]),
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
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.utils.class_weight import compute_class_weight
    
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
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train, y_train)
    models['Neural Network'] = nn_model
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return metrics"""
    results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        results[model_name] = metrics
        print(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        print("")
    
    return results

def save_models(models, label_encoders, scaler, numerical_features, categorical_features):
    """Save all models and preprocessors"""
    import os
    import pickle
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save each model
    for model_name, model in models.items():
        model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} model to {model_filename}")
    
    # Save preprocessors
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    features_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    with open('models/features_info.pkl', 'wb') as f:
        pickle.dump(features_info, f)
    
    print("Saved preprocessors and feature information")

def main():
    st.set_page_config(
        page_title="Credit Risk Prediction Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
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

    st.markdown('<h1 class="main-header">🏦 End-to-End Credit Risk Analysis System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Complete Data Science Pipeline: From Business Understanding to Model Deployment</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.header("Navigation Menu")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select a section:",
        ["🏠 Business Understanding", "📊 Data Exploration", "🛠️ Data Preprocessing", "🤖 Model Training", "📈 Model Evaluation", "🔮 Risk Prediction", "📋 Model Comparison", "ℹ️ About Project"]
    )
    
    # Generate sample data once at the beginning
    df = create_sample_data()
    
    if page == "🏠 Business Understanding":
        st.markdown('<h2 class="sub-header">Business Understanding & Problem Statement</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Credit Risk Prediction System
        
        **Background and Problem Statement:**
        
        Your data science team is starting a project to develop an intelligent credit risk assessment system for ID/X Partners. The aim is to develop machine learning models to detect loan defaults and minimize financial losses.
        
        This is a time-sensitive problem as financial institutions need to quickly assess credit risk to minimize losses and optimize profitability. The challenge is how to accurately determine if a borrower will repay their loan or default?
        
        **Business Goals:**
        - Minimize financial losses from loan defaults
        - Optimize interest rates based on risk
        - Automate loan approval processes
        - Improve portfolio management
        - Enhance customer experience through faster decisions
        
        **Dataset:**
        Historical credit data from Lending Club (2007-2014) with borrower information, loan characteristics, and outcome data.
        
        **Stakeholders:**
        - Risk managers
        - Loan officers
        - Data scientists
        - Compliance team
        - Executive leadership
        """)
        
        # Business metrics
        st.markdown('<h2 class="sub-header">Key Business Metrics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Default Rate</h3>
                <h2>{:.2%}</h2>
                <p>Portfolio Default Rate</p>
            </div>
            """.format(df['loan_status'].mean()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Dataset Size</h3>
                <h2>{:,}</h2>
                <p>Loan Records</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Loan Size</h3>
                <h2>${:,.0f}</h2>
                <p>Average Loan Amount</p>
            </div>
            """.format(df['loan_amnt'].mean()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg FICO Score</h3>
                <h2>{:.0f}</h2>
                <p>Average Credit Score</p>
            </div>
            """.format(df['fico_score'].mean()), unsafe_allow_html=True)

    elif page == "📊 Data Exploration":
        st.markdown('<h2 class="sub-header">Data Exploration & Visualization</h2>', unsafe_allow_html=True)
        
        st.write("**Dataset Overview**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Default Rate: {df['loan_status'].mean():.2%}")
        
        st.write("**Sample Data:**")
        st.dataframe(df.head())
        
        st.write("**Basic Statistics:**")
        st.table(df.describe())
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Loan Status")
            loan_status_counts = df['loan_status'].value_counts()
            fig1 = px.pie(loan_status_counts, values=loan_status_counts.values, 
                          names=['Fully Paid', 'Charged Off'], title='Loan Status Distribution')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Distribution of Loan Grades")
            grade_distribution = df['grade'].value_counts()
            fig2 = px.bar(x=grade_distribution.index, y=grade_distribution.values,
                          title='Loan Grade Distribution', labels={'x': 'Grade', 'y': 'Count'})
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("FICO Score Distribution by Loan Status")
            fig3 = px.histogram(df, x='fico_score', color='loan_status', 
                                title='FICO Score Distribution', 
                                labels={'fico_score': 'FICO Score', 'loan_status': 'Loan Status'})
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            st.subheader("FICO Score by Grade and Loan Status")
            fig4 = px.box(df, x='grade', y='fico_score', color='loan_status',
                          title='FICO Score by Grade', 
                          labels={'fico_score': 'FICO Score', 'grade': 'Grade'})
            st.plotly_chart(fig4, use_container_width=True)

    elif page == "🛠️ Data Preprocessing":
        st.markdown('<h2 class="sub-header">Data Preprocessing & Feature Engineering</h2>', unsafe_allow_html=True)
        
        st.info("Performing data preprocessing and feature engineering...")
        
        # Show original data characteristics
        st.subheader("Original Data Characteristics")
        st.write(f"Dataset shape: {df.shape}")
        st.write(f"Missing values: {df.isnull().sum().sum()}")
        st.write(f"Data types:\n{df.dtypes}")
        
        # Perform preprocessing
        X_train, X_test, y_train, y_test, label_encoders, scaler, numerical_features, categorical_features = preprocess_data(df)
        
        st.success("Data preprocessing completed successfully!")
        
        st.subheader("Preprocessing Steps Applied:")
        st.markdown("""
        - **Categorical Encoding**: Applied Label Encoding to categorical variables
        - **Numerical Scaling**: Applied StandardScaler to numerical variables
        - **Train-Test Split**: 80-20 split with stratification
        - **Feature Engineering**: Created new features like loan-to-income ratio
        """)
        
        st.subheader("Processed Data Characteristics")
        st.write(f"Training set: {X_train.shape}")
        st.write(f"Test set: {X_test.shape}")
        st.write(f"Numerical features: {numerical_features}")
        st.write(f"Categorical features: {categorical_features}")

    elif page == "🤖 Model Training":
        st.markdown('<h2 class="sub-header">Model Training & Algorithm Implementation</h2>', unsafe_allow_html=True)
        
        if st.button("Train All Models"):
            with st.spinner("Preparing data for training..."):
                X_train, X_test, y_train, y_test, label_encoders, scaler, numerical_features, categorical_features = preprocess_data(df)
            
            with st.spinner("Training models (this may take a few minutes)..."):
                models = train_models(X_train, y_train)
            
            with st.spinner("Saving models and preprocessors..."):
                save_models(models, label_encoders, scaler, numerical_features, categorical_features)
            
            st.success("All models have been successfully trained and saved!")
            
            st.subheader("Models Trained:")
            model_descriptions = {
                "Logistic Regression": "Linear model for binary classification with high interpretability",
                "Random Forest": "Ensemble method using decision trees to reduce overfitting",
                "XGBoost": "Gradient boosting algorithm with high predictive performance",
                "SVM": "Support Vector Machine effective for high-dimensional spaces",
                "Neural Network": "Multi-layer perceptron for complex pattern recognition"
            }
            
            for model_name, description in model_descriptions.items():
                st.write(f"**{model_name}**: {description}")

    elif page == "📈 Model Evaluation":
        st.markdown('<h2 class="sub-header">Model Performance Evaluation</h2>', unsafe_allow_html=True)
        
        st.info("This section evaluates model performance using various metrics. Models need to be trained first.")
        
        # Load models if they exist
        try:
            import os
            import pickle
            
            if os.path.exists('models'):
                models = {}
                model_names = [
                    'Logistic Regression', 
                    'Random Forest', 
                    'XGBoost', 
                    'SVM', 
                    'Neural Network'
                ]
                
                for model_name in model_names:
                    filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
                    if os.path.exists(filename):
                        with open(filename, 'rb') as f:
                            models[model_name] = pickle.load(f)
                
                if models:
                    st.success(f"Loaded {len(models)} trained models!")
                    
                    # Prepare test data
                    X_train, X_test, y_train, y_test, le, s, nf, cf = preprocess_data(df)
                    
                    # Evaluate models
                    results = evaluate_models(models, X_test, y_test)
                    
                    # Display results
                    st.subheader("Model Performance Comparison")
                    
                    # Create dataframe for visualization
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
                    
                    # Visualize metrics
                    st.subheader("Performance Visualization")
                    
                    fig = go.Figure()
                    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                    
                    for model_name in results.keys():
                        values = [results[model_name][metric] for metric in metrics]
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=metrics,
                            fill='toself',
                            name=model_name
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Model Performance Radar Chart"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed metrics breakdown
                    st.subheader("Detailed Performance Metrics")
                    
                    col1, col2 = st.columns(2)
                    model_names = list(results.keys())
                    
                    for i, model_name in enumerate(model_names):
                        metrics = results[model_name]
                        if i % 2 == 0:
                            with col1:
                                st.write(f"**{model_name}**")
                                for metric, value in metrics.items():
                                    st.write(f"- {metric.capitalize()}: {value:.4f}")
                                st.markdown("---")
                        else:
                            with col2:
                                st.write(f"**{model_name}**")
                                for metric, value in metrics.items():
                                    st.write(f"- {metric.capitalize()}: {value:.4f}")
                                st.markdown("---")
                else:
                    st.warning("No saved models found. Please train models in the 'Model Training' section.")
            else:
                st.warning("No models directory found. Please train models first.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please train models first using the 'Model Training' section.")

    elif page == "🔮 Risk Prediction":
        st.markdown('<h2 class="sub-header">Real-Time Credit Risk Prediction</h2>', unsafe_allow_html=True)
        
        st.info("Load trained models and make predictions on new loan applications.")
        
        # Load models if they exist
        try:
            import os
            import pickle
            
            if os.path.exists('models'):
                models = {}
                model_names = [
                    'Logistic Regression', 
                    'Random Forest', 
                    'XGBoost', 
                    'SVM', 
                    'Neural Network'
                ]
                
                for model_name in model_names:
                    filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
                    if os.path.exists(filename):
                        with open(filename, 'rb') as f:
                            models[model_name] = pickle.load(f)
                
                if models:
                    st.success("Models loaded successfully! Ready for predictions.")
                    
                    st.info("Enter loan application details below to get risk predictions:")
                    
                    # Input form for prediction
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=1000)
                        int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
                        fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=680, step=10)
                        emp_length = st.slider("Employment Length (Years)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
                    
                    with col2:
                        annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=70000, step=5000)
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
                    
                    if st.button("Calculate Risk Prediction"):
                        st.info("Making predictions with all models...")
                        
                        # Prepare input data
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
                        
                        # Encode categorical variables
                        for col in ['grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state']:
                            if col in input_data.columns:
                                try:
                                    input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
                                except KeyError:
                                    # If category not seen during training, use first category
                                    input_data[col] = 0
                        
                        # Scale numerical features
                        numerical_cols = [col for col in input_data.columns if col in numerical_features]
                        input_scaled = input_data.copy()
                        input_scaled[numerical_cols] = scaler.transform(input_data[numerical_cols])
                        
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
                        
                        # Determine risk level
                        if avg_probability < 0.3:
                            risk_level = "Low Risk"
                            risk_color = "#27ae60"
                            bg_color = "#e8f5e8"
                        elif avg_probability < 0.6:
                            risk_level = "Medium Risk"
                            risk_color = "#f39c12"
                            bg_color = "#fff8e1"
                        else:
                            risk_level = "High Risk"
                            risk_color = "#e74c3c"
                            bg_color = "#ffebee"
                        
                        # Display prediction summary
                        st.markdown(f'''
                        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color}; margin: 20px 0;">
                            <h3 style="color: black;">Risk Assessment Result</h3>
                            <p><strong>Average Default Probability:</strong> {avg_probability:.3f} ({avg_probability*100:.1f}%)</p>
                            <p><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold; font-size: 1.2em;">{risk_level}</span></p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Display individual model predictions
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
                                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                        <h4 style="margin-top: 0; color: {color};">{model_name}</h4>
                                        <p><strong>Default Probability:</strong> {result['probability']:.3f} ({result['probability']*100:.1f}%)</p>
                                        <p><strong>Prediction:</strong> <span style="font-weight: bold; color: {color};">{pred_text}</span></p>
                                        <p><strong>Risk Level:</strong> <span style="color: {'red' if result['risk_level'] == 'High' else 'orange' if result['risk_level'] == 'Medium' else 'green'};">{result['risk_level']}</span></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                with col2:
                                    st.markdown(f"""
                                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
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
                            st.write("This application shows low probability of default based on financial indicators.")
                        elif avg_probability < 0.6:
                            st.warning("⚠️ MEDIUM RISK - Consider additional verification or higher interest rate")
                            st.write("Moderate risk of default - additional scrutiny recommended.")
                        else:
                            st.error("❌ HIGH RISK - Recommend rejection or significantly higher interest rate")
                            st.write("High probability of default - proceed with caution or decline.")
                else:
                    st.warning("No saved models found. Please train models in the 'Model Training' section.")
            else:
                st.warning("No models directory found. Please train models first.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please make sure models are trained and saved first.")

    elif page == "📋 Model Comparison":
        st.markdown('<h2 class="sub-header">Model Registry & Comparison Analysis</h2>', unsafe_allow_html=True)
        
        st.info("Compare the performance of different machine learning models.")
        
        # Load models if they exist
        try:
            import os
            import pickle
            
            if os.path.exists('models'):
                models = {}
                model_names = [
                    'Logistic Regression', 
                    'Random Forest', 
                    'XGBoost', 
                    'SVM', 
                    'Neural Network'
                ]
                
                for model_name in model_names:
                    filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
                    if os.path.exists(filename):
                        with open(filename, 'rb') as f:
                            models[model_name] = pickle.load(f)
                
                if models:
                    st.success(f"Model Registry: {len(models)} models available")
                    
                    # Prepare test data
                    X_train, X_test, y_train, y_test, le, s, nf, cf = preprocess_data(df)
                    
                    # Evaluate all models to get fresh metrics
                    comparison_results = {}
                    for model_name, model in models.items():
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        comparison_results[model_name] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred),
                            'roc_auc': roc_auc_score(y_test, y_pred_proba)
                        }
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame(comparison_results).T
                    
                    # Model ranking based on ROC-AUC
                    ranking = comparison_df.sort_values('roc_auc', ascending=False)
                    
                    st.subheader("Model Performance Ranking (by ROC-AUC)")
                    st.table(ranking.style.format("{:.4f}"))
                    
                    # Visualization
                    st.subheader("Performance Comparison Visualizations")
                    
                    # ROC Curves comparison
                    fig = go.Figure()
                    
                    for model_name in models.keys():
                        y_pred_proba = models[model_name].predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        
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
                        name='Random Classifier',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    fig.update_layout(
                        title='ROC Curves Comparison',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance comparison (for models that support it)
                    st.subheader("Feature Importance Comparison")
                    
                    # Check which models have feature importance
                    fig_feat = go.Figure()
                    
                    if hasattr(models.get('Random Forest', None), 'feature_importances_'):
                        rf_importance = models['Random Forest'].feature_importances_
                        fig_feat.add_trace(go.Bar(
                            x=[f'Feature {i}' for i in range(len(rf_importance))],
                            y=rf_importance,
                            name='Random Forest',
                            marker_color='lightblue'
                        ))
                    
                    if hasattr(models.get('XGBoost', None), 'feature_importances_'):
                        xgb_importance = models['XGBoost'].feature_importances_
                        fig_feat.add_trace(go.Bar(
                            x=[f'Feature {i}' for i in range(len(xgb_importance))],
                            y=xgb_importance,
                            name='XGBoost',
                            marker_color='lightgreen'
                        ))
                    
                    if hasattr(models.get('Logistic Regression', None), 'coef_'):
                        lr_importance = np.abs(models['Logistic Regression'].coef_[0])
                        fig_feat.add_trace(go.Bar(
                            x=[f'Feature {i}' for i in range(len(lr_importance))],
                            y=lr_importance,
                            name='Logistic Regression',
                            marker_color='lightcoral'
                        ))
                    
                    fig_feat.update_layout(
                        title='Feature Importance Comparison',
                        xaxis_title='Feature Index',
                        yaxis_title='Importance',
                        hovermode='x'
                    )
                    
                    st.plotly_chart(fig_feat, use_container_width=True)
                    
                    # Model characteristics
                    st.subheader("Model Characteristics")
                    
                    characteristics = pd.DataFrame({
                        'Model': list(models.keys()),
                        'Algorithm Type': ['Linear', 'Ensemble', 'Boosting', 'SVM', 'Neural Network'],
                        'Interpretability': ['High', 'Medium', 'Medium', 'Low', 'Low'],
                        'Training Speed': ['Fast', 'Medium', 'Medium', 'Slow', 'Slow'],
                        'Overfitting Risk': ['Low', 'Low', 'Medium', 'High', 'High'],
                        'ROC-AUC': [comparison_results[m]['roc_auc'] for m in models.keys()]
                    })
                    
                    characteristics = characteristics.sort_values('ROC-AUC', ascending=False)
                    st.table(characteristics)
                    
                    # Business recommendation
                    best_model = characteristics.iloc[0]['Model']
                    st.info(f"Based on ROC-AUC performance, the {best_model} model is the top performer, but consider your specific business needs (interpretability, speed) when selecting the final model for deployment.")
                else:
                    st.warning("No saved models found. Please train models in the 'Model Training' section.")
            else:
                st.warning("No models directory found. Please train models first.")
        except Exception as e:
            st.error(f"Error loading models for comparison: {str(e)}")

    elif page == "ℹ️ About Project":
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