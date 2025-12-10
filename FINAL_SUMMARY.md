# Rakamin-VIX-Intership-IDX - Credit Risk Prediction Project

## Project Overview
This project implements a comprehensive credit risk prediction system using machine learning techniques. The system follows the CRISP-DM methodology to solve the business problem of loan default prediction (churn analysis) for lending institutions.

## Project Components

### 1. Data Understanding and Analytics
- **File**: `notebooks/01_statistics_and_data_analytics.ipynb`
- Descriptive statistics and exploratory data analysis
- Probability distributions and statistical analysis
- Understanding the Lending Club dataset characteristics

### 2. Cloud Computing and Big Data
- **File**: `notebooks/02_cloud_computing_and_big_data.ipynb`
- Cloud computing for data science implementations
- Big data concepts and implementations
- Infrastructure considerations

### 3. Programming Languages and SQL
- **File**: `notebooks/03_python_and_sql_for_data_science.ipynb`
- Python for data science applications
- SQL querying for data extraction
- Code examples and best practices

### 4. Machine Learning Algorithms
- **File**: `notebooks/04_machine_learning_algorithms_comparison.ipynb`
- Implementation of 5 different ML algorithms:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine
  - Neural Network

### 5. Data Preprocessing and Engineering
- **File**: `notebooks/05_data_cleaning_feature_engineering_imbalance.ipynb`
- Data cleaning techniques
- Feature engineering methods
- Handling imbalanced datasets

### 6. Data Visualization
- **File**: `notebooks/06_data_visualization.ipynb`
- Comprehensive data visualizations using matplotlib, pandas, and seaborn
- Feature relationships and insights
- Dashboard-ready visualizations

### 7. Churn (Default) Prediction Models
- **File**: `notebooks/07_churn_prediction_models.ipynb`
- Specialized models for predicting loan defaults
- Focus on churn prediction methodologies

### 8. End-to-End ML Pipeline
- **File**: `notebooks/08_end_to_end_credit_risk_prediction.ipynb`
- Complete end-to-end pipeline implementation
- All components integrated into single workflow

## Dashboard Implementation
- **File**: `dashboard/app.py`
- Interactive Streamlit dashboard
- Real-time credit risk prediction
- Model comparison and evaluation

## Model Implementation Files
- `src/train_multi_models.py`: Code for training the 5 different ML models
- `src/production_credit_risk_model.py`: Production-ready model implementation
- Various model files in the `models/` directory

## Key Features

### Machine Learning Models
1. **Logistic Regression**: Interpretable linear model
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: High-performance gradient boosting
4. **Support Vector Machine**: Effective for high-dimensional spaces
5. **Neural Network**: Deep learning approach for complex patterns

### Dashboard Features
- Real-time risk assessment for loan applications
- Model performance comparison
- Feature importance visualization
- Risk-based business recommendations
- Interactive data exploration

## Business Impact
- Reduces financial losses from loan defaults
- Improves loan approval decision-making process
- Provides risk-based pricing insights
- Enables automated credit risk assessment

## Technical Implementation
- Python 3.9+ with scikit-learn, pandas, numpy
- Streamlit for interactive dashboard
- Multiple visualization libraries (matplotlib, seaborn, plotly)
- Proper data preprocessing and feature engineering
- Model evaluation and comparison frameworks

## Repository Structure
```
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 01_statistics_and_data_analytics.ipynb
│   ├── 02_cloud_computing_and_big_data.ipynb
│   ├── 03_python_and_sql_for_data_science.ipynb
│   ├── 04_machine_learning_algorithms_comparison.ipynb
│   ├── 05_data_cleaning_feature_engineering_imbalance.ipynb
│   ├── 06_data_visualization.ipynb
│   ├── 07_churn_prediction_models.ipynb
│   └── 08_end_to_end_credit_risk_prediction.ipynb
├── src/                       # Source code modules
│   ├── train_multi_models.py
│   └── production_credit_risk_model.py
├── dashboard/                 # Dashboard implementation
│   └── app.py
├── models/                    # Trained model files (not in Git due to size)
├── Data/                      # Raw data (excluded from Git due to size)
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── documentation/             # Project documentation
```

## Requirements
- Run `pip install -r requirements.txt` to install dependencies
- Python 3.9+ recommended

## Usage
1. Install dependencies
2. Run the Streamlit dashboard: `streamlit run dashboard/app.py`
3. Access the dashboard at `http://localhost:8501`

## Note
Large data files are excluded from this repository due to GitHub's 100MB file size limit. The models use synthetic data generation for demonstration purposes.