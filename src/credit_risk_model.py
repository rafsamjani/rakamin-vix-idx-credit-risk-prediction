"""
Credit Risk Prediction Model
============================

This module implements a comprehensive credit risk prediction system for loan applications.
It includes data preprocessing, feature engineering, model training, and evaluation.

Author: Rafsamjani Anugrah
Date: 2024
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import joblib
import json

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer

# XGBoost
import xgboost as xgb

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# SHAP for interpretability
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CreditRiskModel:
    """
    Comprehensive credit risk prediction model with preprocessing,
    feature engineering, and multiple algorithm support.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the credit risk model.

        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = 'loan_status'
        self.metrics = {}
        self.feature_importance = None

    def _get_default_config(self) -> Dict:
        """Get default configuration for the model."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.1,
            'smote_ratio': 'auto',
            'cv_folds': 5,
            'scoring': 'roc_auc'
        }

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file with basic validation.

        Args:
            filepath: Path to the CSV data file

        Returns:
            Loaded pandas DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)

            if df.empty:
                raise ValueError("Loaded data is empty")

            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            return df

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Invalid data file: {str(e)}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing pipeline.

        Args:
            df: Raw input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")

        # Make a copy to avoid modifying original data
        df_processed = df.copy()

        # 1. Filter for completed loans only
        completed_loans = ['Fully Paid', 'Charged Off']
        df_processed = df_processed[df_processed[self.target_column].isin(completed_loans)]
        logger.info(f"Filtered to {len(df_processed)} completed loans")

        # 2. Convert target variable to binary
        df_processed[self.target_column] = df_processed[self.target_column].map({
            'Fully Paid': 0,
            'Charged Off': 1
        })

        # 3. Handle missing values
        df_processed = self._handle_missing_values(df_processed)

        # 4. Clean numeric columns
        df_processed = self._clean_numeric_columns(df_processed)

        # 5. Create derived features
        df_processed = self._create_derived_features(df_processed)

        # 6. Handle outliers
        df_processed = self._handle_outliers(df_processed)

        logger.info(f"Preprocessing completed. Final shape: {df_processed.shape}")
        return df_processed

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")

        missing_info = df.isnull().sum()
        missing_percent = (missing_info / len(df)) * 100

        # Log missing value information
        for col, (count, percent) in enumerate(zip(missing_info, missing_percent)):
            if count > 0:
                logger.info(f"{missing_info.index[col]}: {count} ({percent:.1f}%) missing")

        # Handle missing values by column type and importance
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            if col in ['emp_title', 'title']:
                # Fill text columns with 'Unknown'
                df[col] = df[col].fillna('Unknown')

            elif col in ['emp_length']:
                # Fill employment length with median by income bracket
                df[col] = df[col].fillna('5+ years')  # Conservative estimate

            elif col in ['revol_util', 'dti']:
                # Fill financial ratios with median
                df[col] = df[col].fillna(df[col].median())

            elif col in ['mths_since_last_delinq', 'mths_since_last_record']:
                # Create flags for missing credit history
                flag_col = f'{col}_missing'
                df[flag_col] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)

            elif df[col].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                df[col] = df[col].fillna(df[col].median())

            else:
                # Fill categorical columns with mode
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert numeric columns."""
        logger.info("Cleaning numeric columns...")

        # Convert percentage columns
        percentage_cols = ['int_rate', 'revol_util']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').astype(float)

        # Clean FICO score columns
        fico_cols = [col for col in df.columns if 'fico' in col.lower()]
        for col in fico_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert date columns
        date_cols = ['issue_d', 'earliest_cr_line']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%b-%y', errors='coerce')

        # Convert employment length to numeric
        if 'emp_length' in df.columns:
            emp_length_mapping = {
                '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            df['emp_length_numeric'] = df['emp_length'].map(emp_length_mapping)
            df['emp_length_numeric'] = df['emp_length_numeric'].fillna(5)  # Median

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for better model performance."""
        logger.info("Creating derived features...")

        # Credit score features
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
            df['fico_score_category'] = pd.cut(
                df['fico_avg'],
                bins=[660, 700, 750, 800, 850],
                labels=['Fair', 'Good', 'Very Good', 'Excellent']
            )

        # Financial ratios
        if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] / 12)
            df['monthly_debt_burden'] = df.get('installment', 0) / (df['annual_inc'] / 12)

        # Effective DTI (including new loan payment)
        if 'dti' in df.columns and 'installment' in df.columns and 'annual_inc' in df.columns:
            monthly_income = df['annual_inc'] / 12
            new_payment_dti = (df['installment'] / monthly_income) * 100
            df['effective_dti'] = df['dti'] + new_payment_dti

        # Credit history features
        if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:
            df['credit_age_years'] = (
                df['issue_d'] - df['earliest_cr_line']
            ).dt.days / 365.25
            df['credit_age_years'] = df['credit_age_years'].fillna(0)

        # Employment stability score
        if 'home_ownership' in df.columns and 'emp_length_numeric' in df.columns:
            df['employment_stability'] = (
                df['emp_length_numeric'] * 7 +
                (df['home_ownership'].isin(['MORTGAGE', 'OWN']) * 20) +
                (df.get('verification_status', '').isin(['Source Verified', 'Verified']) * 13)
            )

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        logger.info("Handling outliers...")

        # Cap extreme values for key financial variables
        caps = {
            'annual_inc': (20000, 500000),  # Income range
            'loan_amnt': (1000, 35000),     # Loan amount limits
            'dti': (0, 40),                 # DTI ratio
            'revol_util': (0, 100)          # Revolving utilization
        }

        for col, (lower, upper) in caps.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        logger.info("Preparing features for modeling...")

        # Select features based on data availability and importance
        key_features = [
            # Loan characteristics
            'loan_amnt', 'term', 'int_rate', 'installment',
            'grade', 'sub_grade', 'purpose',

            # Borrower information
            'annual_inc', 'emp_length_numeric', 'home_ownership',
            'verification_status', 'dti',

            # Credit history
            'fico_avg', 'credit_age_years', 'delinq_2yrs',
            'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',

            # Derived features
            'loan_to_income_ratio', 'effective_dti', 'employment_stability'
        ]

        # Filter available features
        available_features = [f for f in key_features if f in df.columns]
        self.feature_columns = available_features

        X = df[available_features].copy()
        y = df[self.target_column]

        logger.info(f"Selected {len(available_features)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def encode_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features and scale numerical features.

        Args:
            X: Feature DataFrame
            fit: Whether to fit encoders (True for training, False for prediction)

        Returns:
            Encoded DataFrame
        """
        if fit:
            self.preprocessor = {}

        X_encoded = X.copy()

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Encode categorical variables
        for col in categorical_cols:
            if fit:
                if col in ['grade', 'sub_grade']:
                    # Ordinal encoding for loan grades
                    if col == 'grade':
                        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
                    else:  # sub_grade
                        mapping = {}
                        for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                            for i in range(1, 6):
                                mapping[f"{grade}{i}"] = "ABCDEFG".index(grade) * 5 + i

                    self.preprocessor[f'{col}_mapping'] = mapping
                    X_encoded[col] = X[col].map(mapping).fillna(0)

                else:
                    # One-hot encoding for other categorical variables
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    encoded = encoder.fit_transform(X[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    X_encoded = pd.concat([X_encoded.drop(col, axis=1), encoded_df], axis=1)

                    self.preprocessor[f'{col}_encoder'] = encoder
                    self.preprocessor[f'{col}_features'] = feature_names
            else:
                # Use fitted encoders for prediction
                if f'{col}_mapping' in self.preprocessor:
                    mapping = self.preprocessor[f'{col}_mapping']
                    X_encoded[col] = X[col].map(mapping).fillna(0)
                elif f'{col}_encoder' in self.preprocessor:
                    encoder = self.preprocessor[f'{col}_encoder']
                    feature_names = self.preprocessor[f'{col}_features']

                    encoded = encoder.transform(X[[col]])
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    X_encoded = pd.concat([X_encoded.drop(col, axis=1), encoded_df], axis=1)

        # Scale numerical features
        if fit:
            scaler = StandardScaler()
            numerical_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
            X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
            self.preprocessor['scaler'] = scaler
            self.preprocessor['numerical_cols'] = numerical_cols
        else:
            scaler = self.preprocessor['scaler']
            numerical_cols = self.preprocessor['numerical_cols']
            X_encoded[numerical_cols] = scaler.transform(X_encoded[numerical_cols])

        return X_encoded

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models and compare their performance.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        # Encode features
        X_train_encoded = self.encode_features(X_train, fit=True)
        X_test_encoded = self.encode_features(X_test, fit=False)

        # Handle class imbalance with SMOTE
        logger.info("Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=self.config['random_state'])
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

        logger.info(f"Original training set shape: {X_train_encoded.shape}")
        logger.info(f"Resampled training set shape: {X_train_resampled.shape}")
        logger.info(f"Resampled target distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        # Define models to train
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.config['random_state'], max_iter=1000),
            'Random Forest': RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state']
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                random_state=self.config['random_state']
            )
        }

        # Train and evaluate each model
        results = {}
        best_model = None
        best_score = 0

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Train model
            model.fit(X_train_resampled, y_train_resampled)

            # Predictions
            y_pred = model.predict(X_test_encoded)
            y_proba = model.predict_proba(X_test_encoded)[:, 1]

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba
            }

            logger.info(f"{name} - ROC-AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1']:.3f}")

            # Track best model (based on ROC-AUC)
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model

        # Store the best model
        self.model = best_model
        self.metrics = results

        # Calculate feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_test_encoded.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

        logger.info(f"Best model selected with ROC-AUC: {best_score:.3f}")

        return results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            X: Feature DataFrame for prediction

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")

        # Prepare features
        X_encoded = self.encode_features(X, fit=False)

        # Ensure all required columns are present
        missing_cols = set(X_encoded.columns) - set(self.encode_features(pd.DataFrame(columns=self.feature_columns), fit=False).columns)
        if missing_cols:
            logger.warning(f"Missing columns in prediction data: {missing_cols}")

        # Make predictions
        predictions = self.model.predict(X_encoded)
        probabilities = self.model.predict_proba(X_encoded)[:, 1]

        return predictions, probabilities

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with detailed evaluation results
        """
        logger.info("Evaluating model...")

        # Make predictions
        X_test_encoded = self.encode_features(X_test, fit=False)
        y_pred = self.model.predict(X_test_encoded)
        y_proba = self.model.predict_proba(X_test_encoded)[:, 1]

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_summary': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'classification_report': class_report,
            'roc_curve': {
                'false_positive_rate': fpr.tolist(),
                'true_positive_rate': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        }

        return evaluation_results

    def save_model(self, filepath: str) -> None:
        """Save trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model and preprocessing components."""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.metrics = model_data.get('metrics', {})
        self.feature_importance = model_data.get('feature_importance', None)

        logger.info(f"Model loaded from {filepath}")

    def generate_shap_explanations(self, X: pd.DataFrame, sample_size: int = 100) -> Dict:
        """
        Generate SHAP explanations for model interpretability.

        Args:
            X: Sample data for explanation
            sample_size: Number of samples to explain

        Returns:
            Dictionary with SHAP explanations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")

        # Prepare data
        X_encoded = self.encode_features(X, fit=False)
        sample_data = X_encoded.sample(min(sample_size, len(X_encoded)), random_state=42)

        # Create SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.LinearExplainer(self.model, sample_data)

        # Calculate SHAP values
        shap_values = explainer.shap_values(sample_data)

        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': X_encoded.columns.tolist(),
            'sample_data': sample_data
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform cross-validation to assess model stability.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Cross-validation results
        """
        logger.info("Performing cross-validation...")

        # Encode features
        X_encoded = self.encode_features(X, fit=True)

        # Perform cross-validation
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )

        scores = cross_val_score(
            self.model,
            X_encoded,
            y,
            cv=cv,
            scoring=self.config['scoring']
        )

        cv_results = {
            'scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'confidence_interval': {
                'lower': scores.mean() - 1.96 * scores.std(),
                'upper': scores.mean() + 1.96 * scores.std()
            }
        }

        logger.info(f"Cross-validation completed. Mean ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

        return cv_results


def main():
    """
    Main function to run the credit risk modeling pipeline.
    """
    # Initialize model
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1
    }

    model = CreditRiskModel(config)

    try:
        # Load data (assuming data is in data/raw directory)
        df = model.load_data('data/raw/loan_data_2007_2014.csv')

        # Preprocess data
        df_processed = model.preprocess_data(df)

        # Prepare features
        X, y = model.prepare_features(df_processed)

        # Train models
        results = model.train_models(X, y)

        # Display results
        logger.info("\n" + "="*50)
        logger.info("MODEL TRAINING RESULTS")
        logger.info("="*50)

        for model_name, result in results.items():
            metrics = result['metrics']
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall:    {metrics['recall']:.3f}")
            logger.info(f"  F1-Score:  {metrics['f1']:.3f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")

        # Save the best model
        model.save_model('models/best_model.pkl')

        # Save feature importance
        if model.feature_importance is not None:
            model.feature_importance.to_csv('models/feature_importance.csv', index=False)

        logger.info("\nModel training completed successfully!")
        logger.info("Best model saved to models/best_model.pkl")

    except Exception as e:
        logger.error(f"Error in modeling pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()