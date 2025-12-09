# Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-green.svg)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Intelligent Credit Risk Assessment System for ID/X Partners**

## 📋 Project Overview

This comprehensive credit risk prediction system was developed as the final project for the Rakamin Virtual Internship Experience (VIX) Program in partnership with ID/X Partners. The system uses machine learning to predict loan default probability, helping lenders make informed decisions while minimizing financial losses.

## 🎯 Business Problem

**Client**: ID/X Partners (Lending Company)
**Challenge**: How to accurately determine if a borrower will repay their loan or default?
**Impact**: Poor credit decisions can lead to significant financial losses
**Dataset**: Lending Club loan data (2007-2014) with borrower information and outcomes

## 🚀 Key Features

### 🔍 Individual Risk Assessment
- Real-time credit risk evaluation for loan applications
- Comprehensive borrower profile analysis
- Risk score calculation (0-100%)
- Automated approval/rejection recommendations

### 📈 Portfolio Analytics
- Risk distribution analysis
- Default rate visualization
- Performance metrics by loan categories
- Historical trend analysis

### 🎯 Machine Learning Models
- **XGBoost**: Primary prediction model
- **Random Forest**: Ensemble baseline
- **Logistic Regression**: Interpretable benchmark
- **SMOTE**: Handles class imbalance effectively

### 📊 Interactive Dashboard
- User-friendly Streamlit interface
- Real-time risk calculations
- Visual analytics and insights
- Business impact analysis

## 📁 Project Structure

```
credit-risk-prediction/
├── data/                    # Data files
│   ├── raw/                # Original dataset
│   ├── processed/          # Cleaned and processed data
│   └── external/           # Additional reference data
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── credit_risk_analysis.ipynb
├── src/                    # Modular Python code
│   └── credit_risk_model.py
├── models/                 # Trained models
│   ├── credit_risk_model_best.pkl
│   └── feature_importance.csv
├── dashboard/              # Streamlit application
│   ├── app.py
│   └── requirements.txt
├── docs/                   # Technical documentation
│   ├── 01_project_overview.md
│   ├── 02_data_dictionary.md
│   ├── 03_eda_report.md
│   ├── 04_modeling_approach.md
│   ├── 05_model_evaluation.md
│   ├── 06_visualization_guide.md
│   └── 07_deployment_guide.md
├── reports/                # Final reports and presentations
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🛠️ Technology Stack

### Core Technologies
- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

### Dashboard Framework
- **Frontend**: Streamlit (for simplicity and speed)
- **Backend**: Python with ML models
- **Deployment**: Streamlit Cloud/AWS/Azure

### Machine Learning
- **Algorithms**: XGBoost, Random Forest, Logistic Regression
- **Preprocessing**: StandardScaler, OneHotEncoder
- **Imbalance Handling**: SMOTE, SMOTETomek
- **Evaluation**: ROC-AUC, Precision, Recall, F1-Score

## 📊 Model Performance

Our best model (XGBoost) achieves outstanding performance:

| Metric | Score | Benchmark |
|--------|-------|-----------|
| **ROC-AUC** | 0.85+ | > 0.80 ✅ |
| **Accuracy** | 0.80+ | > 0.75 ✅ |
| **Precision** | 0.75+ | > 0.70 ✅ |
| **Recall** | 0.70+ | > 0.65 ✅ |
| **F1-Score** | 0.72+ | > 0.68 ✅ |

### Business Impact
- **Positive Financial Impact**: Optimizes lending decisions
- **Risk Reduction**: Identifies high-risk applications effectively
- **Improved Approval Process**: Automated decision-making for low-risk applications

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### 1. Clone Repository
```bash
git clone https://github.com/rafsamjani/rakamin-vix-idx-credit-risk-prediction.git
cd rakamin-vix-idx-credit-risk-prediction
```

### 2. Create Virtual Environment
```bash
# Run the setup script (recommended)
python setup_environment.py

# Or create manually
python -m venv credit-risk-env

# Activate (Windows)
credit-risk-env\Scripts\activate

# Activate (Linux/Mac)
source credit-risk-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Analysis
```bash
# Start Jupyter Notebook for data analysis
jupyter notebook

# Open notebooks/01_data_exploration.ipynb to begin
```

### 5. Launch Dashboard
```bash
# Navigate to dashboard directory
cd dashboard

# Install dashboard requirements
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### 6. Access the Dashboard
Open your browser and navigate to: `http://localhost:8501`

## 📊 Data Analysis Workflow

### Phase 1: Data Exploration (01_data_exploration.ipynb)
- Load and understand the dataset structure
- Analyze missing values and data quality issues
- Explore target variable distribution
- Initial insights and observations

### Phase 2: Data Cleaning (02_data_cleaning.ipynb)
- Filter completed loans (Fully Paid vs Charged Off)
- Handle missing values with appropriate strategies
- Convert data types (percentages, dates, currencies)
- Create derived features
- Handle outliers

### Phase 3: Feature Engineering & Modeling (03_feature_engineering.ipynb)
- Feature selection based on importance
- Data preprocessing for ML models
- Handle class imbalance with SMOTE
- Train multiple algorithms
- Hyperparameter tuning
- Model evaluation and selection

### Phase 4: Deployment
- Interactive Streamlit dashboard
- Real-time risk assessment
- Portfolio analytics
- Business impact analysis

## 🎯 Key Findings

### Top Risk Factors
1. **FICO Score**: Strongest negative correlation with default
2. **Debt-to-Income Ratio**: Higher DTI indicates higher risk
3. **Employment Length**: Shorter employment history increases risk
4. **Credit Utilization**: High revolving utilization is a red flag
5. **Recent Delinquencies**: Past behavior predicts future defaults

### Risk Profiles
**Low Risk Borrower:**
- FICO score: 750+
- DTI ratio: < 15%
- Employment: 5+ years
- Home ownership: Mortgage or own

**High Risk Borrower:**
- FICO score: < 680
- DTI ratio: > 30%
- Employment: < 2 years
- Recent delinquencies

## 📈 Business Recommendations

### Risk-Based Pricing
- Implement tiered interest rates based on risk scores
- Higher rates for high-risk applicants
- Competitive rates for low-risk borrowers

### Underwriting Guidelines
- Minimum FICO score: 680 for standard loans
- Maximum DTI ratio: 40%
- Minimum employment: 6 months verified
- Income verification required for all applications

### Portfolio Management
- Diversify by loan grade and purpose
- Monitor concentration risk
- Implement early warning systems
- Regular portfolio performance review

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Documentation

Comprehensive documentation is available in the `/docs` folder:

- [Project Overview](docs/01_project_overview.md)
- [Data Dictionary](docs/02_data_dictionary.md)
- [EDA Report](docs/03_eda_report.md)
- [Modeling Approach](docs/04_modeling_approach.md)
- [Model Evaluation](docs/05_model_evaluation.md)
- [Visualization Guide](docs/06_visualization_guide.md)
- [Deployment Guide](docs/07_deployment_guide.md)

## 🏆 Achievements

### Technical Excellence
- ✅ Comprehensive data preprocessing pipeline
- ✅ Advanced feature engineering
- ✅ Multiple ML algorithms with hyperparameter tuning
- ✅ Class imbalance handling
- ✅ Model interpretability with SHAP values

### Business Value
- ✅ Positive financial impact projection
- ✅ Actionable risk assessment tool
- ✅ Automated decision support
- ✅ Real-time risk evaluation
- ✅ Portfolio analytics

### Learning Outcomes
- ✅ End-to-end ML project experience
- ✅ Business acumen in credit risk
- ✅ Data visualization and dashboarding
- ✅ Model deployment considerations

## 📞 Contact

**Author**: Rafsamjani Anugrah
**Email**: [your-email@example.com]
**LinkedIn**: [linkedin.com/in/yourprofile]
**GitHub**: [github.com/rafsamjani]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Rakamin Academy** for the VIX Program opportunity
- **ID/X Partners** for the business problem and mentorship
- **Lending Club** for providing the dataset
- **Open Source Community** for the amazing tools and libraries

---

## 🚀 Deployment

For production deployment instructions, see [Deployment Guide](docs/07_deployment_guide.md).

### Cloud Deployment Options
1. **Streamlit Cloud** (Easiest)
2. **AWS EC2 + Streamlit**
3. **Google Cloud Platform**
4. **Azure App Service**
5. **Docker + Kubernetes**

### Monitoring and Maintenance
- Model performance monitoring
- Data drift detection
- Automated retraining pipeline
- A/B testing for model updates

---

*Last Updated: December 2024*
*Project Status: ✅ Complete*
*Ready for Production: ✅ Yes*