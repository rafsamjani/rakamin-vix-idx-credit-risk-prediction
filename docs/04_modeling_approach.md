# Modeling Approach and Methodology

## Executive Summary

This document outlines the comprehensive modeling approach for developing an accurate credit risk prediction system. The methodology combines traditional machine learning techniques with business domain expertise to create a robust and interpretable model.

## Problem Formulation

### Business Objective
Predict probability of loan default to enable informed lending decisions and minimize financial losses.

### Machine Learning Task
- **Type**: Binary Classification
- **Target**: Loan status (Fully Paid = 0, Charged Off = 1)
- **Goal**: Maximize prediction accuracy while maintaining interpretability
- **Success Metric**: Combination of AUC-ROC, precision, recall, and business impact

### Key Challenges
1. **Class Imbalance**: Majority fully paid loans (~85%) vs. minority defaults (~15%)
2. **Interpretability Requirements**: Business stakeholders need model explanations
3. **Regulatory Compliance**: Fair lending requirements and model transparency
4. **Temporal Dynamics**: Economic conditions affect default patterns

## Data Preprocessing Strategy

### Data Cleaning Pipeline
```python
def preprocess_data(df):
    # 1. Filter completed loans only
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # 2. Handle missing values
    # - Numerical: median imputation for financial ratios
    # - Categorical: mode imputation for demographics
    # - Special: create flags for missing delinquency dates

    # 3. Outlier treatment
    # - Income: cap at 99th percentile
    # - DTI: clip at reasonable bounds
    # - Loan amounts: validate against income

    # 4. Feature engineering
    # - Create derived variables
    # - Encode categorical variables
    # - Standardize numerical features

    return processed_df
```

### Missing Value Handling Strategy

**High Missing (> 50%) Features:**
- `mths_since_last_delinq`: Create binary flag (has_delinquency_history)
- `mths_since_last_record`: Create binary flag (has_public_records)

**Moderate Missing (10-50%) Features:**
- `emp_length`: Median imputation by income bracket
- `revol_util`: Median imputation by credit utilization tier
- `annual_inc_joint`: Fill with individual income for single applications

**Low Missing (< 10%) Features:**
- `dti`: Median imputation
- `title`: Mode imputation or use 'purpose' category

### Outlier Detection and Treatment

**Income Outliers:**
- Method: IQR with 3*IQR upper bound
- Treatment: Cap at 99th percentile
- Business Logic: Income verification required for extreme values

**DTI Outliers:**
- Natural limit: 40% (dataset constraint)
- Treatment: No additional capping needed

**Credit Score Outliers:**
- Minimum threshold: 660 (already filtered)
- Treatment: No outliers within valid range

## Feature Engineering Strategy

### Core Derived Features
```python
def create_derived_features(df):
    # Credit score features
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['fico_score_category'] = pd.cut(df['fico_avg'],
                                     bins=[660, 700, 750, 800, 850],
                                     labels=['Fair', 'Good', 'Very Good', 'Excellent'])

    # Financial ratios
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] / 12)
    df['monthly_debt_burden'] = df['installment'] / (df['annual_inc'] / 12)
    df['effective_dti'] = df['dti'] + (df['installment'] / (df['annual_inc'] / 12) * 100)

    # Credit history features
    df['credit_age_years'] = (pd.to_datetime(df['issue_d']) -
                             pd.to_datetime(df['earliest_cr_line'])).dt.days / 365.25

    # Employment stability features
    df['emp_length_numeric'] = df['emp_length'].replace({
        '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    })

    return df
```

### Risk Score Combinations
```python
def create_risk_scores(df):
    # Credit quality score (0-100)
    df['credit_quality_score'] = (
        (df['fico_avg'] - 660) / 190 * 40 +  # FICO contribution
        (df['credit_age_years'] / 30) * 20 +   # Credit age contribution
        (100 - df['revol_util']) * 0.15 +      # Credit utilization
        max(0, (100 - df['dti'] * 2.5)) * 0.25 # DTI contribution
    )

    # Employment stability score
    df['employment_stability_score'] = (
        df['emp_length_numeric'] * 7 +         # Employment length
        (df['home_ownership'].isin(['MORTGAGE', 'OWN']) * 20) +  # Home ownership
        (df['verification_status'].isin(['Source Verified', 'Verified']) * 13)  # Income verification
    )

    return df
```

### Categorical Encoding Strategy

**Ordinal Variables:**
- `grade`: A=1, B=2, ..., G=7 (natural ordering)
- `emp_length`: 0-10 years (numeric mapping)
- `sub_grade`: Grade * 10 + subgrade number

**Nominal Variables (One-Hot Encoding):**
- `home_ownership`: RENT, MORTGAGE, OWN, OTHER
- `purpose`: debt_consolidation, credit_card, etc.
- `addr_state`: Group by default rate regions

**High-Cardinality Variables:**
- `emp_title`: Group by occupation categories
- `title`: Use purpose as proxy

## Model Selection Strategy

### Baseline Models
1. **Logistic Regression**: Interpretable baseline
2. **Decision Tree**: Simple non-linear model
3. **Random Forest**: Ensemble baseline
4. **XGBoost**: Performance benchmark

### Advanced Models
1. **LightGBM**: Efficient gradient boosting
2. **CatBoost**: Automatic categorical handling
3. **Neural Network**: For complex patterns
4. **Ensemble**: Weighted combination of top models

### Model Comparison Framework
```python
model_comparison = {
    'Logistic Regression': {
        'pros': ['Highly interpretable', 'Fast training', 'Good baseline'],
        'cons': ['Linear assumptions', 'Limited complexity'],
        'use_case': 'Business explanation and quick baseline'
    },
    'Random Forest': {
        'pros': ['Handles non-linearity', 'Feature importance', 'Robust'],
        'cons': ['Less interpretable', 'Slower prediction'],
        'use_case': 'Baseline non-linear model'
    },
    'XGBoost': {
        'pros': ['High performance', 'Handles missing values', 'Regularization'],
        'cons': ['Complex tuning', 'Black-box tendency'],
        'use_case': 'Primary predictive model'
    }
}
```

## Class Imbalance Handling

### Evaluation Metrics Priority
```python
evaluation_metrics = {
    'primary': 'ROC_AUC',           # Overall discrimination
    'business_critical': {
        'precision': 'Minimize false positives (good customers rejected)',
        'recall': 'Maximize true positives (catch defaults)',
        'f1_score': 'Balance precision and recall'
    },
    'secondary': ['accuracy', 'confusion_matrix', 'classification_report']
}
```

### Imbalance Treatment Techniques

1. **Class Weighting**
```python
class_weights = {0: 1, 1: 5}  # Penalize minority class more
model.fit(X_train, y_train, class_weight=class_weights)
```

2. **SMOTE (Synthetic Minority Oversampling)**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

3. **Combined Approach (SMOTE + Tomek Links)**
```python
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
```

## Hyperparameter Optimization Strategy

### Grid Search Framework
```python
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', {0: 1, 1: 3}, {0: 1, 1: 5}]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [1, 3, 5]  # Handle imbalance
    }
}
```

### Cross-Validation Strategy
- **Type**: Stratified 5-fold CV
- **Metric**: ROC-AUC primary, F1 secondary
- **Validation**: Time-based split for temporal validation
- **Early Stopping**: Prevent overfitting for boosting models

## Model Evaluation Framework

### Business Metrics
```python
def business_metrics(y_true, y_pred, y_proba):
    # Confusion matrix based costs
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Business cost assumptions
    cost_false_positive = 5000   # Lost good customer profit
    cost_false_negative = 25000  # Default loss
    benefit_true_positive = 5000 # Saved from default
    benefit_true_negative = 5000 # Profit from good customer

    # Calculate net financial impact
    total_cost = (fp * cost_false_positive + fn * cost_false_negative)
    total_benefit = (tp * benefit_true_positive + tn * benefit_true_negative)

    return total_benefit - total_cost
```

### Performance Thresholds
- **Minimum Acceptable**: ROC-AUC > 0.80
- **Target Performance**: ROC-AUC > 0.85
- **Business Impact**: Positive net financial gain

### Model Validation Approach
1. **Temporal Validation**: Train on older data, test on recent
2. **Cross-Validation**: Ensure robustness
3. **Business Simulation**: Test with different economic scenarios
4. **A/B Testing**: Compare against current underwriting process

## Feature Selection Strategy

### Correlation Analysis
- Remove features with > 0.8 correlation
- Use variance inflation factor (VIF) for multicollinearity
- Keep business-critical features despite correlation

### Feature Importance Methods
1. **Tree-based importance**: Random Forest, XGBoost
2. **Permutation importance**: Model-agnostic
3. **SHAP values**: Local and global interpretability
4. **Business domain expertise**: Manual feature selection

### Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=RandomForestClassifier(),
               step=1,
               cv=StratifiedKFold(5),
               scoring='roc_auc')
rfecv.fit(X_train, y_train)
```

## Model Interpretability Strategy

### SHAP (SHapley Additive exPlanations)
```python
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
```

### Business Rule Extraction
- Convert model decisions to understandable rules
- Identify key risk thresholds
- Create manual underwriting guidelines

## Deployment Considerations

### Model Monitoring
- **Performance Drift**: Regular AUC monitoring
- **Feature Drift**: Input distribution changes
- **Business Impact**: Financial outcome tracking
- **Data Quality**: Missing value patterns

### Retraining Strategy
- **Frequency**: Monthly or quarterly
- **Trigger**: Performance degradation or new data availability
- **Validation**: Backtesting with holdout periods

### Model Versioning
- Track model parameters and performance
- Maintain audit trail for regulatory compliance
- Enable rollback capability

## Risk Management

### Model Risk Assessment
1. **Accuracy Risk**: Model may make incorrect predictions
2. **Bias Risk**: Model may discriminate against protected groups
3. **Stability Risk**: Performance may degrade over time
4. **Interpretability Risk**: Model decisions may be hard to explain

### Mitigation Strategies
- **Accuracy**: Ensemble methods, extensive validation
- **Bias**: Fairness metrics, protected group analysis
- **Stability**: Regular monitoring and retraining
- **Interpretability**: SHAP values, feature importance

## Success Criteria

### Technical Success
- ROC-AUC > 0.85 on test set
- Positive business financial impact
- Stable performance over time
- Acceptable prediction speed (< 1 second)

### Business Success
- Reduced default rate by target percentage
- Improved loan approval efficiency
- Better risk-adjusted pricing
- Regulatory compliance maintained

### Implementation Success
- Successful integration with existing systems
- User acceptance and adoption
- Scalable to future data volumes
- Maintainable and updateable codebase

This comprehensive modeling approach ensures development of an accurate, interpretable, and business-relevant credit risk prediction system that meets both technical and regulatory requirements.