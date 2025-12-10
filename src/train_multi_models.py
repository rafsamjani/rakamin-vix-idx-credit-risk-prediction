import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
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
        print()
    
    return results

def save_models(models, label_encoders, scaler, numerical_features, categorical_features):
    """Save all models and preprocessors"""
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

def load_models():
    """Load all models and preprocessors"""
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
    
    # Load preprocessors
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/features_info.pkl', 'rb') as f:
        features_info = pickle.load(f)
    
    return models, label_encoders, scaler, features_info

def make_prediction(input_data, models, label_encoders, scaler, features_info):
    """Make predictions for a single sample across all models"""
    # Process the input data
    df = pd.DataFrame([input_data])
    
    # Add engineered features
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['interest_cost'] = df['loan_amnt'] * (df['int_rate'] / 100)
    df['installment_to_income_ratio'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    
    # Encode categorical variables
    for col in ['grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state']:
        if col in df.columns and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                # Handle unseen categories by using the first category
                df[col] = 0
    
    # Scale numerical features
    numerical_features = features_info['numerical_features']
    df_scaled = df.copy()
    df_scaled[numerical_features] = scaler.transform(df[numerical_features])
    
    # Make predictions with all models
    results = {}
    for model_name, model in models.items():
        pred_proba = model.predict_proba(df_scaled)[0, 1]
        pred_class = model.predict(df_scaled)[0]
        results[model_name] = {'probability': pred_proba, 'prediction': pred_class}
    
    return results

def main():
    print("Starting Credit Risk Prediction Project - End to End")
    print("="*60)
    
    # Step 1: Data Preparation
    print("Step 1: Loading and preparing data...")
    df = create_sample_data()
    print(f"Data loaded with shape: {df.shape}")
    print(f"Default rate: {df['loan_status'].mean():.2%}")
    
    # Step 2: Preprocessing
    print("\nStep 2: Preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoders, scaler, numerical_features, categorical_features = preprocess_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 3: Model Training
    print("\nStep 3: Training models...")
    models = train_models(X_train, y_train)
    
    # Step 4: Model Evaluation
    print("Step 4: Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Step 5: Save Models
    print("Step 5: Saving models...")
    save_models(models, label_encoders, scaler, numerical_features, categorical_features)
    
    # Step 6: Test with sample input
    print("Step 6: Testing with sample input...")
    sample_input = {
        'loan_amnt': 15000,
        'int_rate': 12.0,
        'installment': 400,
        'annual_inc': 70000,
        'dti': 15.0,
        'fico_score': 700,
        'emp_length': 5.0,
        'grade': 'B',
        'home_ownership': 'MORTGAGE',
        'verification_status': 'Verified',
        'purpose': 'debt_consolidation',
        'addr_state': 'CA'
    }
    
    try:
        loaded_models, loaded_encoders, loaded_scaler, loaded_features = load_models()
        predictions = make_prediction(sample_input, loaded_models, loaded_encoders, loaded_scaler, loaded_features)
        
        print(f"\nPredictions for the sample input:")
        for model_name, result in predictions.items():
            prob = result['probability']
            pred = result['prediction']
            risk_level = 'HIGH' if prob > 0.5 else 'LOW'
            print(f"{model_name}: Probability = {prob:.3f}, Prediction = {pred} (Risk: {risk_level})")
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    print("\n" + "="*60)
    print("Credit Risk Prediction Project - Complete!")
    print("All models have been trained, evaluated, and saved to the models/ folder.")
    print("The system is now ready for deployment in the Streamlit dashboard.")
    
    # Return key components for dashboard integration
    return {
        'models': models,
        'results': results,
        'df': df,
        'X_test': X_test,
        'y_test': y_test
    }

if __name__ == "__main__":
    project_data = main()