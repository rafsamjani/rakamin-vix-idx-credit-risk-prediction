"""
Test System for Credit Risk Model
==================================

This script tests the complete credit risk prediction system to ensure
everything is working correctly before deployment.

Author: Rafsamjani Anugrah
Date: December 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings

# Fix encoding for Windows
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'English_United States.1252')
    except:
        pass

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.production_credit_risk_model import CreditRiskModel, CreditRiskAPI, create_sample_application

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def test_model_loading():
    """Test if the model can be loaded successfully."""
    print("\n" + "="*50)
    print("TEST 1: Model Loading")
    print("="*50)

    try:
        model = CreditRiskModel()

        if model.is_loaded:
            print("[OK] Model loaded successfully")
            print(f"   Model type: {model.best_model_name}")
            print(f"   Training date: {model.model_metadata.get('training_date', 'Unknown')}")
            print(f"   Dataset shape: {model.model_metadata.get('dataset_shape', 'Unknown')}")
            print(f"   Optimal threshold: {model.optimal_threshold}")
            return True
        else:
            print("[FAIL] Model failed to load")
            return False

    except Exception as e:
        print(f"[FAIL] Error loading model: {str(e)}")
        return False


def test_single_prediction():
    """Test single prediction functionality."""
    print("\n" + "="*50)
    print("TEST 2: Single Prediction")
    print("="*50)

    try:
        model = CreditRiskModel()

        # Test with low risk application
        low_risk_app = create_sample_application()
        low_risk_app['fico_range_low'] = 750
        low_risk_app['annual_inc'] = 150000
        low_risk_app['dti'] = 10

        result = model.predict(low_risk_app)

        print("✅ Low risk application prediction:")
        print(f"   Risk Score: {result['risk_score']:.1f}%")
        print(f"   Decision: {result['decision']}")
        print(f"   Risk Category: {result['risk_category']}")

        # Test with high risk application
        high_risk_app = create_sample_application()
        high_risk_app['fico_range_low'] = 600
        high_risk_app['annual_inc'] = 30000
        high_risk_app['dti'] = 40

        result = model.predict(high_risk_app)

        print("\n✅ High risk application prediction:")
        print(f"   Risk Score: {result['risk_score']:.1f}%")
        print(f"   Decision: {result['decision']}")
        print(f"   Risk Category: {result['risk_category']}")

        return True

    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        return False


def test_batch_prediction():
    """Test batch prediction functionality."""
    print("\n" + "="*50)
    print("TEST 3: Batch Prediction")
    print("="*50)

    try:
        model = CreditRiskModel()

        # Create test batch
        batch_data = []
        for i in range(5):
            app = create_sample_application()
            app['id'] = f'TEST_{i+1}'
            app['annual_inc'] = 50000 + (i * 10000)
            app['fico_range_low'] = 650 + (i * 10)
            batch_data.append(app)

        df_batch = pd.DataFrame(batch_data)
        results = model.batch_predict(df_batch)

        print("✅ Batch prediction completed successfully")
        print(f"   Processed {len(results)} applications")
        print("\n   Results summary:")
        for _, row in results.iterrows():
            print(f"   - {row['id']}: Risk Score {row['risk_score']:.1f}% - {row['decision']}")

        return True

    except Exception as e:
        print(f"❌ Error with batch prediction: {str(e)}")
        return False


def test_api_wrapper():
    """Test the API wrapper functionality."""
    print("\n" + "="*50)
    print("TEST 4: API Wrapper")
    print("="*50)

    try:
        api = CreditRiskAPI()
        sample_app = create_sample_application()
        sample_app['id'] = 'API_TEST_001'

        result = api.assess_single_application(sample_app)

        print("✅ API wrapper test successful")
        print(f"   Application ID: {result['application_id']}")
        print(f"   Assessment Date: {result['assessment_date']}")
        print(f"   Recommendation: {result['recommendation']['action']}")
        print(f"   Confidence Level: {result['recommendation']['confidence']}")

        return True

    except Exception as e:
        print(f"❌ Error with API wrapper: {str(e)}")
        return False


def test_feature_importance():
    """Test feature importance loading."""
    print("\n" + "="*50)
    print("TEST 5: Feature Importance")
    print("="*50)

    try:
        model = CreditRiskModel()
        importance_df = model.get_feature_importance(top_n=10)

        if importance_df is not None:
            print("✅ Feature importance loaded successfully")
            print("\n   Top 10 Important Features:")
            for _, row in importance_df.iterrows():
                print(f"   - {row['feature']}: {row['importance']:.4f}")
            return True
        else:
            print("⚠️ Feature importance not available (this is expected for some model types)")
            return True

    except Exception as e:
        print(f"❌ Error loading feature importance: {str(e)}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*50)
    print("TEST 6: Edge Cases & Error Handling")
    print("="*50)

    try:
        model = CreditRiskModel()

        # Test with missing values
        incomplete_app = {
            'loan_amnt': 10000,
            'annual_inc': 50000
            # Missing many required fields
        }

        result = model.predict(incomplete_app)
        print("✅ Handled incomplete application")
        print(f"   Risk Score: {result['risk_score']:.1f}%")

        # Test with extreme values
        extreme_app = create_sample_application()
        extreme_app['annual_inc'] = 1000000  # Very high income
        extreme_app['loan_amnt'] = 1000  # Very small loan

        result = model.predict(extreme_app)
        print("\n✅ Handled extreme values")
        print(f"   Risk Score: {result['risk_score']:.1f}%")

        # Test different thresholds
        sample_app = create_sample_application()
        for threshold in [0.3, 0.5, 0.7]:
            result = model.predict(sample_app, threshold=threshold)
            print(f"\n   Threshold {threshold}: Decision = {result['decision']}")

        return True

    except Exception as e:
        print(f"❌ Error with edge cases: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "="*60)
    print("CREDIT RISK MODEL - SYSTEM TEST SUITE")
    print("="*60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Model Loading", test_model_loading),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("API Wrapper", test_api_wrapper),
        ("Feature Importance", test_feature_importance),
        ("Edge Cases", test_edge_cases)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues before deployment.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)