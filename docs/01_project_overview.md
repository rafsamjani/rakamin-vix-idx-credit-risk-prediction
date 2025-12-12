# Credit Risk Prediction Project Overview

## Project Information
- **Project Name**: Credit Risk Prediction System
- **Company**: ID/X Partners
- **Program**: Rakamin Academy Virtual Internship Experience (VIX)
- **Author**: Rafsamjani Anugrah
- **Repository**: [rakamin-vix-idx-credit-risk-prediction](https://github.com/rafsamjani/rakamin-vix-idx-credit-risk-prediction)

## Business Context

### Client Background
ID/X Partners is a lending company that provides loans to individuals and businesses. As part of their risk management strategy, they need to assess the creditworthiness of potential borrowers to minimize financial losses from defaults.

### Problem Statement
The primary business challenge is to develop an accurate credit risk assessment system that can predict whether a borrower will repay their loan or default. This prediction helps the company:
- Minimize financial losses from bad loans
- Make informed lending decisions
- Optimize interest rates based on risk levels
- Improve overall portfolio performance

### Dataset Information
- **Source**: Lending Club loan data (2007-2014)
- **Location**: `Dataset/raw/loan_data_2007_2014.csv` (240MB, ~870K rows)
- **Data Dictionary**: `Dataset/LCDataDictionary.xlsx` (Complete feature definitions)
- **Processed Data**: `Dataset/processed/` (Cleaned and engineered features)
- **Target Variable**: Loan status (Fully Paid vs Charged Off)
- **Key Features**: Loan amount, interest rate, FICO score, DTI, annual income, etc.

## Project Objectives

### Primary Objective
Build a machine learning model that predicts loan default probability with high accuracy and interpretability.

### Secondary Objectives
1. Identify key risk factors that contribute to loan defaults
2. Create an interactive dashboard for risk assessment
3. Provide actionable insights for business decision-making
4. Develop a scalable solution for real-time predictions

## Success Metrics

### Technical Metrics
- **Accuracy**: > 85% overall prediction accuracy
- **Precision**: > 80% for default prediction (minimize false positives)
- **Recall**: > 75% for default prediction (capture most defaults)
- **AUC-ROC**: > 0.85
- **F1-Score**: > 0.78

### Business Metrics
- **Loss Reduction**: Minimize false negatives (missed defaults)
- **Profit Optimization**: Balance risk assessment with business growth
- **Processing Speed**: < 1 second prediction time
- **Model Interpretability**: Clear explanation of risk factors

## Project Timeline

### Week 1: Data Exploration & Preprocessing (2 days)
- Import and explore dataset
- Understand data dictionary
- Identify data quality issues
- Initial EDA and visualization

### Week 2: Data Processing & Feature Engineering (3 days)
- Data cleaning and formatting
- Handle missing values and outliers
- Feature engineering and selection
- Data standardization

### Week 3: Model Development (4 days)
- Train multiple algorithms
- Handle class imbalance
- Hyperparameter tuning
- Model selection and validation

### Week 4: Deployment & Documentation (5 days)
- Build interactive dashboard
- Create comprehensive documentation
- Prepare final presentation
- Testing and validation

## Technical Stack

### Core Technologies
- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

### Dashboard Framework
- **Primary**: Streamlit (for simplicity and speed)
- **Alternative**: Plotly Dash (if needed for complex interactions)

### Version Control
- **Repository**: GitHub
- **Branch Strategy**: GitFlow (main, develop, feature branches)

## Expected Deliverables

1. **Python Script** (`credit_risk_model.py`)
   - Complete data processing pipeline
   - Model training and evaluation
   - Prediction functionality

2. **Jupyter Notebook** (`credit_risk_analysis.ipynb`)
   - Step-by-step analysis
   - Code explanations and visualizations
   - Results interpretation

3. **Interactive Dashboard** (`dashboard/app.py`)
   - Real-time prediction interface
   - Risk visualization tools
   - Business insights

4. **Documentation** (docs/ folder)
   - Technical documentation
   - Data dictionary
   - Deployment guide

5. **Presentation** (reports/final_presentation.pdf)
   - Project overview
   - Key findings
   - Business recommendations

## Risk Assessment & Mitigation

### Technical Risks
- **Data Quality**: Missing values, inconsistent formats
  - *Mitigation*: Comprehensive data cleaning pipeline
- **Class Imbalance**: Fewer defaults than fully paid loans
  - *Mitigation*: SMOTE, class weighting, ensemble methods
- **Overfitting**: Model may not generalize to new data
  - *Mitigation*: Cross-validation, regularization, feature selection

### Business Risks
- **Model Interpretability**: Black box models may be hard to explain
  - *Mitigation*: Use SHAP values, feature importance, simpler models as baseline
- **Regulatory Compliance**: Fair lending requirements
  - *Mitigation*: Monitor for bias, ensure model transparency

## Project Structure

```
Rakamin-VIX-Intership-IDX/
├── Dataset/                # Data files (actual location)
│   ├── raw/               # Original dataset
│   │   └── loan_data_2007_2014.csv    # Main dataset (240MB)
│   ├── processed/         # Cleaned and processed data
│   └── LCDataDictionary.xlsx          # Data dictionary
├── notebooks/             # Jupyter notebooks for analysis
│   └── 05_end_to_end_credit_risk_prediction.ipynb
├── src/                   # Modular Python code
│   └── production_credit_risk_model.py
├── models/                # Trained model files
│   ├── credit_risk_model_v2.pkl
│   └── neural_network_model.h5
├── dashboard/             # Streamlit application
│   ├── dashboard_v2.py
│   └── main.py
├── reports/               # Final reports and presentations
├── docs/                  # Technical documentation
│   └── 01_project_overview.md
├── requirements.txt       # Python dependencies
├── PROJECT_DOCUMENTATION.md
├── INSTALLATION_GUIDE.md
└── README.md             # Project documentation
```

## Dataset Details

### Main Dataset
- **File**: `Dataset/raw/loan_data_2007_2014.csv`
- **Size**: 240MB, approximately 870,000 rows
- **Period**: 2007-2014 loan data
- **Source**: Lending Club

### Data Dictionary
- **File**: `Dataset/LCDataDictionary.xlsx`
- **Purpose**: Complete definition of all 75+ features
- **Usage**: Feature selection and engineering reference

### Key Features Available
1. **Loan Information**: loan_amnt, funded_amnt, term, int_rate, installment
2. **Borrower Information**: annual_inc, emp_length, home_ownership, purpose
3. **Credit History**: fico_range_low, fico_range_high, dti, revol_util
4. **Employment**: emp_length, verification_status
5. **Geographic**: addr_state, zip_code

### Target Variable
- **Primary**: `loan_status` (Fully Paid, Charged Off, Current, Late, etc.)
- **Binary**: `target` (0 = Fully Paid, 1 = Charged Off)

## Next Steps

1. ✅ Set up development environment (venv + requirements)
2. ✅ Load and explore the dataset (`Dataset/raw/loan_data_2007_2014.csv`)
3. ✅ Understand business requirements through data dictionary (`Dataset/LCDataDictionary.xlsx`)
4. ✅ Complete exploratory data analysis (in notebook)
5. ✅ Develop machine learning and deep learning models
6. 🔄 Build interactive dashboard (`dashboard/dashboard_v2.py`)
7. 🔄 Finalize documentation and deployment