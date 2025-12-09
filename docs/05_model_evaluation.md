# Model Evaluation and Performance Metrics

## Executive Summary

This document outlines comprehensive evaluation framework for assessing credit risk prediction models, including technical metrics, business impact analysis, and model selection criteria.

## Evaluation Objectives

### Primary Goals
1. **Technical Performance**: Measure predictive accuracy and robustness
2. **Business Impact**: Quantify financial benefits and risks
3. **Model Comparison**: Select optimal model for production
4. **Risk Assessment**: Identify model limitations and failure modes

### Evaluation Scope
- Cross-validation performance
- Holdout test set validation
- Business scenario analysis
- Sensitivity analysis
- Fairness and bias assessment

## Performance Metrics Framework

### Primary Technical Metrics

#### 1. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
```python
from sklearn.metrics import roc_auc_score, roc_curve

# Calculation
auc_score = roc_auc_score(y_test, y_proba)

# Interpretation
auc_interpretation = {
    0.5: "Random guessing",
    0.6: "Poor discrimination",
    0.7: "Acceptable discrimination",
    0.8: "Good discrimination",
    0.9: "Excellent discrimination"
}
```

**Business Significance:**
- Measures ability to rank-order risk
- Threshold-independent evaluation
- Standard industry benchmark

#### 2. Precision and Recall
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision: Minimize false positives (good customers rejected)
precision = precision_score(y_test, y_pred)

# Recall: Maximize true positives (catch defaults)
recall = recall_score(y_test, y_pred)

# F1 Score: Balance precision and recall
f1 = f1_score(y_test, y_pred)
```

**Business Impact:**
- **High Precision**: Fewer good customers denied loans
- **High Recall**: More defaulters identified and rejected
- **Trade-off**: Optimization based on business priorities

#### 3. Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as seaborn
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

# Business interpretation
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives: {tn} (Good loans approved)")
print(f"False Positives: {fp} (Good loans rejected - opportunity cost)")
print(f"False Negatives: {fn} (Bad loans approved - financial loss)")
print(f"True Positives: {tp} (Bad loans rejected - risk avoided)")
```

### Business-Specific Metrics

#### 1. Financial Impact Score
```python
def calculate_financial_impact(y_true, y_pred, loan_amounts):
    """
    Calculate the net financial impact of model decisions

    Args:
        y_true: Actual loan outcomes
        y_pred: Model predictions (1 = reject, 0 = approve)
        loan_amounts: Corresponding loan amounts

    Returns:
        Net financial impact (positive = profit, negative = loss)
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Business assumptions
    avg_interest_rate = 0.15  # 15% average interest
    default_loss_rate = 0.60   # 60% loss on default
    cost_of_capital = 0.05     # 5% cost of funds

    # Calculations
    avg_loan_amount = np.mean(loan_amounts)

    # Benefits
    profit_from_good_loans = tn * avg_loan_amount * avg_interest_rate
    savings_from_avoided_defaults = tp * avg_loan_amount * default_loss_rate

    # Costs
    opportunity_cost = fp * avg_loan_amount * (avg_interest_rate - cost_of_capital)
    default_losses = fn * avg_loan_amount * default_loss_rate

    net_impact = (profit_from_good_loans + savings_from_avoided_defaults) -
                 (opportunity_cost + default_losses)

    return net_impact
```

#### 2. Risk-Adjusted Return Metrics
```python
def calculate_risk_adjusted_metrics(y_true, y_proba, loan_amounts):
    """
    Calculate risk-adjusted performance metrics
    """

    # Expected return calculation
    expected_return = np.mean(y_proba * loan_amounts * 0.15)  # Expected interest income
    expected_loss = np.mean((1-y_proba) * loan_amounts * 0.60)  # Expected loss

    # Sharpe-like ratio for loan portfolio
    risk_adjusted_return = (expected_return - expected_loss) / np.std(y_proba)

    return {
        'expected_return': expected_return,
        'expected_loss': expected_loss,
        'risk_adjusted_return': risk_adjusted_return
    }
```

## Model Comparison Framework

### Performance Ranking System
```python
def rank_models(models_metrics):
    """
    Rank models based on comprehensive evaluation criteria

    Scoring weights (can be adjusted based on business priorities):
    - ROC-AUC: 25%
    - Financial Impact: 30%
    - Precision: 20%
    - Recall: 15%
    - Stability: 10%
    """

    weights = {
        'auc_score': 0.25,
        'financial_impact': 0.30,
        'precision': 0.20,
        'recall': 0.15,
        'stability': 0.10
    }

    # Normalize metrics to 0-1 scale
    normalized_scores = {}
    for model, metrics in models_metrics.items():
        score = 0
        for metric, weight in weights.items():
            # Normalize metric (implement based on metric type)
            normalized = normalize_metric(metrics[metric], metric)
            score += normalized * weight
        normalized_scores[model] = score

    # Rank models
    ranked_models = sorted(normalized_scores.items(),
                          key=lambda x: x[1], reverse=True)

    return ranked_models
```

### Model Selection Criteria
```python
selection_criteria = {
    'minimum_requirements': {
        'roc_auc': 0.80,
        'precision': 0.70,
        'recall': 0.65,
        'financial_impact': 0  # Must be positive
    },
    'preferred_targets': {
        'roc_auc': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'financial_impact': 1000000  # $1M positive impact
    }
}
```

## Validation Strategy

### 1. Cross-Validation Performance
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cross_val_score

def cross_validate_model(model, X, y, cv=5):
    """
    Perform stratified cross-validation with multiple metrics
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    metrics = ['roc_auc', 'precision', 'recall', 'f1']
    results = {}

    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }

    return results
```

### 2. Temporal Validation
```python
def temporal_validation(model, X, y, time_column, train_periods, test_period):
    """
    Validate model performance across different time periods
    """
    temporal_results = {}

    for train_period in train_periods:
        # Split data by time
        train_mask = (time_column >= train_period[0]) & (time_column <= train_period[1])
        test_mask = (time_column >= test_period[0]) & (time_column <= test_period[1])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        temporal_results[f"{train_period[0]}_{train_period[1]}"] = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

    return temporal_results
```

### 3. Stress Testing Scenarios
```python
def stress_test_model(model, X, y, scenarios):
    """
    Test model performance under various economic scenarios
    """
    stress_results = {}

    for scenario_name, scenario_func in scenarios.items():
        # Apply scenario transformation to features
        X_stressed = scenario_func(X.copy())

        # Evaluate performance
        y_pred = model.predict(X_stressed)
        y_proba = model.predict_proba(X_stressed)[:, 1]

        stress_results[scenario_name] = {
            'roc_auc': roc_auc_score(y, y_proba),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred)
        }

    return stress_results

# Example scenarios
def recession_scenario(X):
    """Simulate economic recession conditions"""
    X_stressed = X.copy()
    # Increase DTI by 20%
    X_stressed['dti'] *= 1.2
    # Decrease income by 15%
    X_stressed['annual_inc'] *= 0.85
    # Increase credit utilization
    X_stressed['revol_util'] = np.minimum(100, X_stressed['revol_util'] * 1.3)
    return X_stressed
```

## Threshold Optimization

### Business-Optimal Threshold Selection
```python
def find_optimal_threshold(y_true, y_proba, financial_params):
    """
    Find threshold that maximizes business financial impact

    Args:
        financial_params: dict with cost/benefit parameters
            - cost_false_positive: Opportunity cost
            - cost_false_negative: Default loss
            - benefit_true_positive: Risk avoided
            - benefit_true_negative: Profit from good loan
    """

    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0.5
    best_impact = float('-inf')

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate financial impact
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_impact = (
            tn * financial_params['benefit_true_negative'] +
            tp * financial_params['benefit_true_positive'] -
            fp * financial_params['cost_false_positive'] -
            fn * financial_params['cost_false_negative']
        )

        if total_impact > best_impact:
            best_impact = total_impact
            best_threshold = threshold

    return best_threshold, best_impact
```

## Model Diagnostics

### 1. Calibration Analysis
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def analyze_calibration(y_true, y_proba, n_bins=10):
    """
    Analyze model calibration - probability vs. actual frequency
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )

    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.show()

    # Calculate Brier score
    brier_score = np.mean((y_proba - y_true) ** 2)

    return {
        'brier_score': brier_score,
        'calibration_data': list(zip(mean_predicted_value, fraction_of_positives))
    }
```

### 2. Feature Stability Analysis
```python
def analyze_feature_stability(model, X_train, X_test):
    """
    Analyze how stable feature importance is across train/test sets
    """
    # Get feature importance from training
    train_importance = get_feature_importance(model, X_train, y_train)
    test_importance = get_feature_importance(model, X_test, y_test)

    # Calculate correlation between importances
    importance_correlation = np.corrcoef(train_importance, test_importance)[0, 1]

    return {
        'importance_correlation': importance_correlation,
        'train_importance': train_importance,
        'test_importance': test_importance
    }
```

## Fairness and Bias Assessment

### Protected Group Analysis
```python
def analyze_fairness(model, X, y, protected_attributes):
    """
    Assess model fairness across protected groups

    Args:
        protected_attributes: List of column names for protected groups
            (e.g., race, gender, age if available)
    """

    fairness_results = {}

    for attribute in protected_attributes:
        if attribute in X.columns:
            group_results = {}

            for group in X[attribute].unique():
                mask = X[attribute] == group
                y_group = y[mask]
                y_proba_group = model.predict_proba(X[mask])[:, 1]

                group_results[group] = {
                    'mean_predicted_probability': np.mean(y_proba_group),
                    'actual_default_rate': np.mean(y_group),
                    'sample_size': len(y_group)
                }

            fairness_results[attribute] = group_results

    return fairness_results
```

### Disparate Impact Analysis
```python
def calculate_disparate_impact(y_true, y_pred, protected_groups):
    """
    Calculate disparate impact ratio (80% rule)

    Returns:
        Ratio of approval rates between protected and non-protected groups
        Value < 0.8 indicates potential discrimination
    """

    # Calculate approval rates for each group
    approval_rates = {}
    for group in protected_groups:
        group_mask = protected_groups == group
        approval_rate = np.mean(y_pred[group_mask] == 0)  # 0 = approved
        approval_rates[group] = approval_rate

    # Calculate disparate impact
    if len(approval_rates) == 2:
        groups = list(approval_rates.keys())
        di_ratio = min(approval_rates.values()) / max(approval_rates.values())
    else:
        di_ratio = None

    return {
        'approval_rates': approval_rates,
        'disparate_impact_ratio': di_ratio,
        'passes_80_percent_rule': di_ratio >= 0.8 if di_ratio else None
    }
```

## Performance Reporting

### Model Performance Dashboard
```python
def generate_performance_report(model, X_train, X_test, y_train, y_test):
    """
    Generate comprehensive model performance report
    """

    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    report = {
        'model_type': type(model).__name__,
        'training_performance': {
            'roc_auc': roc_auc_score(y_train, y_train_proba),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1_score': f1_score(y_train, y_train_pred)
        },
        'test_performance': {
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred)
        },
        'overfitting_assessment': {
            'roc_auc_diff': roc_auc_score(y_train, y_train_proba) - roc_auc_score(y_test, y_test_proba),
            'precision_diff': precision_score(y_train, y_train_pred) - precision_score(y_test, y_test_pred),
            'recall_diff': recall_score(y_train, y_train_pred) - recall_score(y_test, y_test_pred)
        }
    }

    return report
```

## Decision Framework

### Model Selection Matrix
```python
def create_model_selection_matrix(models_performance):
    """
    Create decision matrix for model selection
    """

    criteria = {
        'Performance': 0.30,
        'Interpretability': 0.20,
        'Speed': 0.15,
        'Stability': 0.20,
        'Business_Impact': 0.15
    }

    # Score each model (0-10 for each criterion)
    model_scores = {}
    for model_name, performance in models_performance.items():
        scores = {
            'Performance': min(10, performance['roc_auc'] * 10),
            'Interpretability': get_interpretability_score(model_name),
            'Speed': get_speed_score(model_name),
            'Stability': performance['stability_score'],
            'Business_Impact': min(10, performance['financial_impact'] / 1000000)
        }

        # Calculate weighted score
        total_score = sum(scores[crit] * weight for crit, weight in criteria.items())
        model_scores[model_name] = {'scores': scores, 'total_score': total_score}

    return model_scores
```

This comprehensive evaluation framework ensures thorough assessment of model performance from both technical and business perspectives, enabling informed decision-making for model deployment.