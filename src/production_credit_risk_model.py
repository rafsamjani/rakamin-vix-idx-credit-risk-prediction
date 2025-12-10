"""
Credit Risk Prediction Model - Production Ready
=============================================

This module provides production-ready functions for credit risk prediction
using the trained machine learning model.

Author: Rafsamjani Anugrah
Company: ID/X Partners
Date: December 2024
Version: 2.0.0
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Union
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class CreditRiskModel:
    """
    Production-ready credit risk prediction model.

    This class encapsulates all functionality needed for credit risk assessment
    including model loading, preprocessing, prediction, and result interpretation.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the CreditRiskModel.

        Args:
            model_path: Path to the trained model file
        """
        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self.feature_columns = None
        self.best_model_name = None
        self.optimal_threshold = 0.5
        self.is_loaded = False

        if model_path is None:
            # Default model path
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'credit_risk_model_v2.pkl')

        self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load the trained model and associated artifacts.

        Args:
            model_path: Path to the saved model file

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info(f"Loading model from {model_path}")

            # Load the complete model data
            model_data = joblib.load(model_path)

            # Extract components
            self.model = model_data['best_model']
            self.preprocessor = model_data['preprocessor']
            self.feature_columns = model_data['feature_columns']
            self.best_model_name = model_data['best_model_name']
            self.model_metadata = model_data['model_metadata']
            self.optimal_threshold = model_data.get('optimal_threshold', 0.5)

            self.is_loaded = True
            logger.info(f"Model loaded successfully: {self.best_model_name}")
            logger.info(f"Optimal threshold: {self.optimal_threshold}")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False

    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.

        Args:
            input_data: Dictionary or DataFrame with loan application data

        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan
            logger.warning(f"Added missing column: {col}")

        # Select only the columns the model expects
        df = df[self.feature_columns]

        return df

    def predict(self, input_data: Union[Dict, pd.DataFrame],
                threshold: Optional[float] = None) -> Dict:
        """
        Make credit risk prediction.

        Args:
            input_data: Loan application data
            threshold: Decision threshold (uses optimal if not provided)

        Returns:
            Dict: Prediction results including probability and decision
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Please load model first.")

        # Use optimal threshold if not provided
        if threshold is None:
            threshold = self.optimal_threshold

        # Preprocess input
        df_preprocessed = self.preprocess_input(input_data)

        # Make prediction
        try:
            # Get prediction probability
            prediction_proba = self.model.predict_proba(df_preprocessed)[:, 1]

            # Make binary decision
            prediction = (prediction_proba >= threshold).astype(int)

            # Create result dictionary
            result = {
                'prediction': int(prediction[0]),
                'probability': float(prediction_proba[0]),
                'threshold_used': threshold,
                'risk_score': float(prediction_proba[0] * 100),  # As percentage
                'model_used': self.best_model_name,
                'timestamp': datetime.now().isoformat()
            }

            # Add interpretation
            result['decision'] = self._interpret_prediction(result['prediction'],
                                                           result['risk_score'])
            result['risk_category'] = self._get_risk_category(result['risk_score'])

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def _interpret_prediction(self, prediction: int, risk_score: float) -> str:
        """
        Interpret the prediction result.

        Args:
            prediction: Binary prediction (0 or 1)
            risk_score: Risk score percentage

        Returns:
            str: Human-readable decision
        """
        if prediction == 0:
            if risk_score < 15:
                return "APPROVED - Low risk applicant"
            elif risk_score < 25:
                return "APPROVED - Medium risk applicant"
            else:
                return "REVIEW REQUIRED - Consider additional verification"
        else:
            return "REJECTED - High risk applicant"

    def _get_risk_category(self, risk_score: float) -> str:
        """
        Categorize risk score into risk levels.

        Args:
            risk_score: Risk score percentage

        Returns:
            str: Risk category
        """
        if risk_score < 10:
            return "Very Low"
        elif risk_score < 20:
            return "Low"
        elif risk_score < 35:
            return "Medium"
        elif risk_score < 50:
            return "High"
        else:
            return "Very High"

    def batch_predict(self, input_data: pd.DataFrame,
                      threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Make predictions for multiple applications.

        Args:
            input_data: DataFrame with multiple loan applications
            threshold: Decision threshold

        Returns:
            pd.DataFrame: Input data with prediction results added
        """
        results = []

        for _, row in input_data.iterrows():
            result = self.predict(row.to_dict(), threshold)
            results.append(result)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Combine with input data
        output_df = pd.concat([input_data.reset_index(drop=True), results_df], axis=1)

        return output_df

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            pd.DataFrame: Feature importance scores or None if not available
        """
        try:
            # Try to load feature importance
            feature_importance_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'feature_importance.csv'
            )

            if os.path.exists(feature_importance_path):
                importance_df = pd.read_csv(feature_importance_path)
                return importance_df.head(top_n)
            else:
                logger.warning("Feature importance file not found")
                return None

        except Exception as e:
            logger.error(f"Error loading feature importance: {str(e)}")
            return None

    def evaluate_performance(self, y_true: np.ndarray,
                           y_pred_proba: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dict: Performance metrics
        """
        from sklearn.metrics import (accuracy_score, precision_score,
                                   recall_score, f1_score, roc_auc_score,
                                   confusion_matrix)

        # Make predictions using optimal threshold
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'threshold_used': self.optimal_threshold
        }

        # Add confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

        return metrics


class CreditRiskAPI:
    """
    API wrapper for CreditRiskModel for easier integration.
    """

    def __init__(self, model_path: str = None):
        """Initialize the API with a model."""
        self.model = CreditRiskModel(model_path)

    def assess_single_application(self, application_data: Dict) -> Dict:
        """
        Assess a single loan application.

        Args:
            application_data: Dictionary with application details

        Returns:
            Dict: Comprehensive assessment result
        """
        # Make prediction
        prediction_result = self.model.predict(application_data)

        # Add additional information
        result = {
            'application_id': application_data.get('id', 'UNKNOWN'),
            'assessment_date': datetime.now().strftime('%Y-%m-%d'),
            'risk_assessment': prediction_result,
            'recommendation': self._get_recommendation(prediction_result),
            'next_steps': self._get_next_steps(prediction_result['prediction'])
        }

        return result

    def _get_recommendation(self, prediction_result: Dict) -> Dict:
        """
        Get business recommendation based on prediction.

        Args:
            prediction_result: Prediction result from model

        Returns:
            Dict: Business recommendation
        """
        risk_score = prediction_result['risk_score']
        decision = prediction_result['decision']

        if 'APPROVED' in decision:
            if risk_score < 15:
                return {
                    'action': 'APPROVE',
                    'confidence': 'HIGH',
                    'notes': 'Low risk applicant. Standard terms apply.'
                }
            else:
                return {
                    'action': 'APPROVE_WITH_CONDITIONS',
                    'confidence': 'MEDIUM',
                    'notes': 'Consider higher interest rate or additional verification.'
                }
        elif 'REVIEW' in decision:
            return {
                'action': 'MANUAL_REVIEW',
                'confidence': 'LOW',
                'notes': 'Application requires manual underwriter review.'
            }
        else:
            return {
                'action': 'REJECT',
                'confidence': 'HIGH',
                'notes': 'High risk applicant. Application rejected.'
            }

    def _get_next_steps(self, prediction: int) -> list:
        """
        Get next steps based on prediction.

        Args:
            prediction: Binary prediction (0 or 1)

        Returns:
            list: Next steps
        """
        if prediction == 0:
            return [
                'Proceed with loan documentation',
                'Schedule loan disbursement',
                'Set up payment reminders',
                'Monitor account performance'
            ]
        else:
            return [
                'Send rejection notification',
                'Provide feedback on improvement areas',
                'Consider alternative products if applicable',
                'Maintain records for compliance'
            ]


def create_sample_application() -> Dict:
    """
    Create a sample loan application for testing.

    Returns:
        Dict: Sample application data
    """
    return {
        'loan_amnt': 15000,
        'term': '36 months',
        'int_rate': 12.5,
        'installment': 500.0,
        'grade': 'C',
        'sub_grade': 'C1',
        'emp_title': 'Software Engineer',
        'emp_length': '5 years',
        'home_ownership': 'RENT',
        'annual_inc': 75000,
        'verification_status': 'Verified',
        'issue_d': 'Dec-2024',
        'purpose': 'debt_consolidation',
        'title': 'Debt consolidation',
        'zip_code': '945xx',
        'addr_state': 'CA',
        'dti': 20.0,
        'delinq_2yrs': 0,
        'earliest_cr_line': 'Jan-2010',
        'fico_range_low': 680,
        'fico_range_high': 690,
        'inq_last_6mths': 2,
        'mths_since_last_delinq': None,
        'mths_since_last_record': None,
        'open_acc': 10,
        'pub_rec': 0,
        'revol_bal': 5000,
        'revol_util': '30%',
        'total_acc': 25,
        'collections_12_mths_ex_med': 0,
        'mths_since_last_major_derog': None,
        'application_type': 'Individual'
    }


def main():
    """
    Main function for testing the model.
    """
    logger.info("Testing Credit Risk Model")

    # Initialize model
    model = CreditRiskModel()

    if not model.is_loaded:
        logger.error("Failed to load model. Exiting.")
        return

    # Test with sample application
    sample_app = create_sample_application()

    # Make prediction
    result = model.predict(sample_app)

    # Display results
    print("\n" + "="*50)
    print("CREDIT RISK ASSESSMENT RESULT")
    print("="*50)
    print(f"Model Used: {result['model_used']}")
    print(f"Risk Score: {result['risk_score']:.1f}%")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Decision: {result['decision']}")
    print(f"Default Probability: {result['probability']:.2%}")
    print(f"Threshold Used: {result['threshold_used']}")
    print("="*50)

    # Test API wrapper
    print("\nTesting API Wrapper:")
    api = CreditRiskAPI()
    api_result = api.assess_single_application(sample_app)

    print(f"\nRecommendation: {api_result['recommendation']['action']}")
    print(f"Confidence: {api_result['recommendation']['confidence']}")
    print(f"Notes: {api_result['recommendation']['notes']}")

    print("\nNext Steps:")
    for step in api_result['next_steps']:
        print(f"  - {step}")


if __name__ == "__main__":
    main()