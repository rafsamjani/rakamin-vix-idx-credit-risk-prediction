# Credit Risk Prediction - End-to-End Data Science Project

This project demonstrates a complete data science pipeline for credit risk prediction, following the CRISP-DM methodology from business understanding to model deployment.

## Project Overview

The goal of this project is to develop machine learning models to predict loan defaults (churn) for lending institutions. The system uses historical lending data to assess credit risk and help financial institutions make informed lending decisions.

## Project Structure

```
Rakamin-VIX-Intership-IDX/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── src/
│   └── train_multi_models.py # Model training scripts
├── dashboard/
│   └── integrated_dashboard.py # Alternative dashboard implementation
├── models/                   # Saved models and preprocessors
├── notebooks/               # Jupyter notebooks for analysis
└── README.md               # This file
```

## Features

### Business Understanding
- Defined credit risk problem statement
- Identified key business metrics
- Established success criteria

### Data Science Pipeline
- Data exploration and visualization
- Data preprocessing and feature engineering
- Model training with 5 algorithms
- Model evaluation and comparison

### Algorithms Implemented
1. **Logistic Regression**: Linear model with high interpretability
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: Gradient boosting with high performance
4. **Support Vector Machine**: Effective for high-dimensional problems
5. **Neural Network**: Deep learning for complex patterns

### Interactive Dashboard
- End-to-end navigation from business understanding to predictions
- Real-time credit risk assessment
- Model comparison and evaluation
- Model registry and management

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Navigation Guide

The application has the following sections:

- **Project Overview**: Business understanding and objectives
- **Data Exploration**: Dataset analysis and visualization  
- **Data Preprocessing**: Feature engineering and transformation
- **Model Training**: Training interfaces for 5 ML algorithms
- **Model Evaluation**: Performance metrics and comparison
- **Predictions**: Real-time risk assessment
- **Model Registry**: Model management and tracking
- **About**: Project documentation

## Data Preprocessing

The system performs:
- Categorical encoding using LabelEncoder
- Numerical scaling using StandardScaler
- Feature engineering (loan-to-income ratio, interest cost, etc.)
- Train-test split with stratification
- Handling of imbalanced data

## Model Architecture

Each model is trained on the same preprocessed dataset with the following characteristics:

- **Logistic Regression**: L2 regularization, interpretable coefficients
- **Random Forest**: 100 trees with depth control
- **XGBoost**: 100 estimators with early stopping
- **SVM**: RBF kernel with probability estimation
- **Neural Network**: Two hidden layers with dropout

## Business Impact

This system is designed to:

- **Reduce Financial Losses**: By identifying high-risk applicants
- **Optimize Interest Rates**: Based on calculated risk scores
- **Improve Approval Process**: Automation for low-risk applications
- **Enhance Portfolio Management**: Through risk distribution analysis
- **Assist Compliance**: By providing explainable predictions

## Key Metrics

- **ROC-AUC**: Primary metric for model comparison
- **Precision**: Proportion of predicted defaults that are actual defaults
- **Recall**: Proportion of actual defaults correctly identified
- **F1-Score**: Balance between precision and recall

## Deployment

The system is designed for:
- Real-time loan application assessment
- Batch processing of applications
- Model monitoring and management
- Integration with existing lending platforms

## Technical Specifications

- **Frontend**: Streamlit for interactive dashboard
- **ML Frameworks**: scikit-learn, XGBoost, TensorFlow
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Pickle for serialization

## Getting Started

1. Train the models using the "Model Training" section in the dashboard
2. Evaluate performance in the "Model Evaluation" section
3. Use the "Predictions" section for real-time risk assessment
4. Manage models in the "Model Registry" section

## License

This project is created for educational purposes and demonstrates data science best practices.