# Credit Risk Prediction Dashboard

This interactive dashboard predicts loan defaults using 5 different machine learning models, helping financial institutions make informed lending decisions. The system was developed following CRISP-DM methodology as part of the Rakamin VIX Internship program.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard Sections](#dashboard-sections)
- [Models Implemented](#models-implemented)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Business Impact](#business-impact)

## Project Overview

This project analyzes credit risk using a simulated Lending Club dataset to predict whether a borrower will default on their loan (churn). The aim is to develop an intelligent credit risk assessment system that helps financial institutions minimize losses from defaulted loans.

The system compares 5 different machine learning algorithms:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM)
5. Neural Network

## Features

- **Interactive Interface**: Real-time credit risk prediction for loan applications
- **Multiple ML Models**: Comparison of 5 different algorithms
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC
- **Feature Analysis**: Importance visualization for risk factors
- **Risk Assessment**: Categorization of loan applications by risk level
- **Model Comparison**: Side-by-side performance evaluation
- **Business Recommendations**: Actionable insights based on predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credit-risk-dashboard
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. If no requirements.txt exists, install the necessary packages:
```bash
pip install streamlit pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn plotly imbalanced-learn
```

## Usage

1. To run the dashboard:
```bash
streamlit run app.py
```

2. The dashboard will open in your default browser at `http://localhost:8501`

3. Use the sidebar to navigate between different sections of the dashboard

## Dashboard Sections

### 1. Home (🏠)
- Project overview and business context
- Key metrics and findings
- Brief introduction to the credit risk problem

### 2. Data Overview (📊)
- Dataset information and statistics
- Distribution of key variables
- Default rate analysis

### 3. Model Performance (📈)
- Performance metrics comparison between models
- ROC curves visualization
- Confusion matrices

### 4. Feature Analysis (🔍)
- Feature importance for each model
- Correlation analysis
- Key risk factors identification

### 5. Individual Predictions (📋)
- Form to input loan application details
- Real-time risk prediction from all models
- Business recommendation for the application

### 6. Model Comparison (📚)
- Detailed comparison of all 5 models
- Model characteristics and trade-offs
- Recommendation for production use

### 7. About (ℹ️)
- Complete project documentation
- Algorithm explanations
- Business impact analysis

## Models Implemented

### Logistic Regression
- **Strengths**: Interpretable, fast training, provides probability estimates
- **Use Case**: When interpretability is crucial for regulatory compliance

### Random Forest
- **Strengths**: Handles non-linear relationships, feature importance, robust to outliers
- **Use Case**: Good balance between performance and interpretability

### XGBoost
- **Strengths**: High predictive accuracy, handles missing values, regularization
- **Use Case**: When highest accuracy is the primary objective

### Support Vector Machine
- **Strengths**: Effective in high-dimensional spaces, memory efficient
- **Use Case**: When linear separation boundary isn't sufficient

### Neural Network
- **Strengths**: Captures complex non-linear patterns, feature learning
- **Use Case**: When large datasets are available and complexity is needed

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow/Keras
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Preprocessing**: Scikit-learn utilities

## Project Structure

```
├── app.py                    # Main Streamlit dashboard application
├── models/                   # Trained model files
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── svm_model.pkl
│   ├── neural_network_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── features_info.pkl
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── documentation.md          # Detailed usage guide
```

## Business Impact

### For Financial Institutions:
- **Reduce Defaults**: Identify high-risk borrowers before loan approval
- **Optimize Pricing**: Adjust interest rates based on calculated risk
- **Automate Decisions**: Streamline approval processes for low-risk applications
- **Portfolio Management**: Better understanding of risk distribution

### Key Benefits:
- Minimized financial losses from bad loans
- Improved loan approval accuracy
- Faster decision-making process
- Regulatory compliance through model interpretability
- Data-driven risk management

### Risk Categories:
- **Low Risk** (<30% default probability): Consider standard approval
- **Medium Risk** (30-60% default probability): Additional verification recommended
- **High Risk** (>60% default probability): Consider rejection or high interest rate

## Troubleshooting

1. **Dashboard won't launch**: Make sure all dependencies are installed and you're running Python 3.7+

2. **Missing models**: The app will train models automatically if not found, but this may take a few minutes

3. **Performance issues**: The app works best with the default sample size; larger datasets may slow it down

4. **Model accuracy seems low**: The sample data is simulated; real implementations would use actual Lending Club data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## License

This project is created for educational purposes as part of the Rakamin VIX Internship program.

## Author

Data Science Team - Rakamin VIX Internship Program

## Acknowledgments

- Lending Club for making the dataset publicly available
- Streamlit team for the excellent dashboarding framework
- Scikit-learn team for the comprehensive ML library
- The open-source community for all the valuable tools and libraries used in this project