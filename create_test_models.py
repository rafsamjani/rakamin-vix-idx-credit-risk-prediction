import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

print('Creating simple model for testing...')

# Create dummy model data
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(np.random.rand(100, 10), np.random.randint(0, 2, 100))

scaler = StandardScaler()
scaler.fit(np.random.rand(100, 10))

model_data = {
    'best_model': model,
    'best_model_name': 'RandomForest',
    'preprocessor': scaler,
    'feature_columns': [f'feature_{i}' for i in range(10)],
    'numeric_features': [f'feature_{i}' for i in range(10)],
    'categorical_features': [],
    'best_metrics': {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.7, 'f1': 0.72, 'roc_auc': 0.81},
    'optimal_threshold': 0.5,
    'model_metadata': {
        'training_date': '2024-12-10',
        'model_version': '2.0.0-test',
        'dataset_shape': (1000, 10)
    }
}

# Save models directory if doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model_data, 'models/credit_risk_model_v2.pkl')

# Create dummy feature importance
importance_df = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(10)],
    'importance': np.random.rand(10)
})
importance_df.to_csv('models/feature_importance.csv', index=False)

# Create dummy performance summary
perf_df = pd.DataFrame({
    'RandomForest': [0.8, 0.75, 0.7, 0.72, 0.81],
    'LogisticRegression': [0.75, 0.70, 0.68, 0.69, 0.76],
    'XGBoost': [0.82, 0.78, 0.73, 0.75, 0.84]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

perf_df.to_csv('models/model_performance_summary.csv')

print('Mock models created successfully!')