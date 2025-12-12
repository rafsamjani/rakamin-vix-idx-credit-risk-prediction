# ============================================================
# Credit Risk Prediction System - End to End
# ============================================================

# %% [Cell 1] - Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
import os

# %% [Cell 2] - Load and Explore Data
# notebooks/01_data_exploration.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../Dataset/raw/loan_data_2007_2014.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Target variable distribution
print("\nTarget Variable Distribution:")
print(df['loan_status'].value_counts(normalize=True))

# %% [Cell 3] - Data Preprocessing Class
# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_report = {}
        
    def filter_completed_loans(self):
        """Keep only completed loans (Fully Paid and Charged Off)"""
        self.df = self.df[self.df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
        self.cleaning_report['filtered_loans'] = len(self.df)
        return self.df
    
    def create_target_variable(self):
        """Create binary target variable: 0=Fully Paid, 1=Charged Off"""
        self.df['target'] = (self.df['loan_status'] == 'Charged Off').astype(int)
        self.cleaning_report['target_created'] = True
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in key features"""
        # Drop columns with >50% missing values
        missing_threshold = 0.5
        cols_to_drop = self.df.columns[self.df.isnull().mean() > missing_threshold]
        self.df = self.df.drop(columns=cols_to_drop)
        self.cleaning_report['dropped_columns'] = list(cols_to_drop)
        
        # Impute numeric columns with median
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='median')
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
        
        # Impute categorical columns with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        imputer = SimpleImputer(strategy='most_frequent')
        self.df[categorical_cols] = imputer.fit_transform(self.df[categorical_cols])
        
        self.cleaning_report['missing_values_handled'] = True
        return self.df
    
    def convert_data_types(self):
        """Convert data types for better analysis"""
        # Convert percentage strings to floats
        if 'int_rate' in self.df.columns:
            self.df['int_rate'] = self.df['int_rate'].str.replace('%', '').astype(float)
        
        # Convert employment length to numeric
        if 'emp_length' in self.df.columns:
            emp_length_map = {
                '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            self.df['emp_length_numeric'] = self.df['emp_length'].map(emp_length_map)
        
        # Convert term to numeric
        if 'term' in self.df.columns:
            self.df['term_months'] = self.df['term'].str.extract('(\d+)').astype(int)
        
        self.cleaning_report['data_types_converted'] = True
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        return self.cleaning_report

# %% [Cell 4] - Feature Engineering Class
# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_report = {}
        
    def create_derived_features(self):
        """Create new features from existing data"""
        # FICO score average
        if 'fico_range_low' in self.df.columns and 'fico_range_high' in self.df.columns:
            self.df['fico_avg'] = (self.df['fico_range_low'] + self.df['fico_range_high']) / 2
        
        # Loan to income ratio
        if 'loan_amnt' in self.df.columns and 'annual_inc' in self.df.columns:
            self.df['loan_to_income'] = self.df['loan_amnt'] / self.df['annual_inc']
        
        # Debt to income ratio
        if 'dti' in self.df.columns:
            self.df['dti_ratio'] = self.df['dti'] / 100  # Convert percentage to ratio
        
        # Monthly payment to income ratio
        if 'installment' in self.df.columns and 'annual_inc' in self.df.columns:
            self.df['payment_to_income'] = (self.df['installment'] * 12) / self.df['annual_inc']
        
        self.feature_report['derived_features_created'] = True
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop(['loan_status'], errors='ignore')
        
        # One-hot encode categorical variables
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded = encoder.fit_transform(self.df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        
        # Drop original categorical columns and add encoded ones
        self.df = pd.concat([self.df.drop(categorical_cols, axis=1), encoded_df], axis=1)
        
        self.feature_report['categorical_encoded'] = list(categorical_cols)
        return self.df
    
    def scale_features(self):
        """Scale numerical features"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols.drop(['target'], errors='ignore')
        
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        
        self.feature_report['features_scaled'] = list(numeric_cols)
        return self.df
    
    def get_engineered_features(self):
        return self.df
    
    def get_feature_report(self):
        return self.feature_report

# %% [Cell 5] - Model Training Class
# src/model_training.py - Updated with custom directory path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.available_models = []
        
    def prepare_data(self, df, target_col='target'):
        """Prepare data for modeling"""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = model
        self.available_models.append('logistic_regression')
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        self.available_models.append('random_forest')
        return model
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = model
        self.available_models.append('gradient_boosting')
        return model
    
    def train_neural_network(self, hidden_layer_sizes=(100,), activation='relu', 
                          solver='adam', alpha=0.0001, max_iter=500):
        """Train Neural Network using scikit-learn's MLPClassifier"""
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            verbose=False
        )
        model.fit(self.X_train, self.y_train)
        self.models['neural_network'] = model
        self.available_models.append('neural_network')
        return model
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        self.model_metrics[model_name] = {
            'metrics': metrics,
            'model': model
        }
        
        return metrics
    
    def train_all_models(self):
        """Train all available models"""
        print("Training Logistic Regression...")
        lr_model = self.train_logistic_regression()
        self.evaluate_model(lr_model, 'logistic_regression')
        
        print("Training Random Forest...")
        rf_model = self.train_random_forest()
        self.evaluate_model(rf_model, 'random_forest')
        
        print("Training Gradient Boosting...")
        gb_model = self.train_gradient_boosting()
        self.evaluate_model(gb_model, 'gradient_boosting')
        
        print("Training Neural Network...")
        nn_model = self.train_neural_network(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500
        )
        self.evaluate_model(nn_model, 'neural_network')
        
        print("All models trained successfully!")
    
    def find_best_model(self, metric='roc_auc'):
        """Find the best performing model based on specified metric"""
        best_score = -1
        best_model_name = None
        
        for model_name, model_data in self.model_metrics.items():
            if model_name != 'best_model':
                current_score = model_data['metrics'][metric]
                if current_score > best_score:
                    best_score = current_score
                    best_model_name = model_name
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        self.model_metrics['best_model'] = {
            'name': best_model_name,
            'metric': metric,
            'score': best_score
        }
        
        print(f"Best model: {best_model_name} with {metric}: {best_score:.4f}")
        return self.best_model, best_model_name
    
    def get_model_comparison(self):
        """Return comparison of all models"""
        comparison = {}
        for model_name, model_data in self.model_metrics.items():
            if model_name != 'best_model':
                comparison[model_name] = model_data['metrics']
        
        return pd.DataFrame(comparison).T
    
    def save_model(self, model, filename):
        """Save trained model to file with comprehensive error handling"""
        try:
            # Ensure directory exists
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    print(f"✅ Created directory: {directory}")
                except Exception as e:
                    print(f"❌ Error creating directory: {e}")
                    return False
            
            # Validate filename
            if not filename.endswith('.pkl'):
                print(f"⚠️ Warning: Filename doesn't end with .pkl: {filename}")
                filename = filename + '.pkl'
            
            # Save model
            joblib.dump(model, filename)
            print(f"✅ Model saved successfully to: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            print(f"❌ Filename: {filename}")
            print(f"❌ Directory exists: {os.path.exists(directory) if directory else 'N/A'}")
            print(f"❌ Is writable: {os.access(directory, os.W_OK) if directory else 'N/A'}")
            return False
    
    def save_all_models(self, output_dir="D:\\Projek pribadi\\scholarship,exchange,pelatihan\\Rakamin-VIX-Intership-IDX\\models"):
        """Save all trained models to the specified directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Created output directory: {output_dir}")
        
        success_count = 0
        for model_name, model in self.models.items():
            filename = os.path.join(output_dir, f"{model_name}.pkl")
            if self.save_model(model, filename):
                success_count += 1
        
        print(f"✅ Saved {success_count}/{len(self.models)} models successfully!")
        return success_count == len(self.models)
    
    def load_model(self, filename):
        """Load trained model from file"""
        try:
            model = joblib.load(filename)
            print(f"✅ Model loaded successfully from: {filename}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"❌ Filename: {filename}")
            print(f"❌ File exists: {os.path.exists(filename)}")
            return None
    
    def get_available_models(self):
        """Return list of available models"""
        return self.available_models


# %% [Cell 6] - Main Execution
if __name__ == "__main__":
    try:
        print("🚀 Starting Credit Risk Model Training...")
        
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'loan_amnt': np.random.uniform(1000, 35000, n_samples),
            'int_rate': np.random.uniform(5, 25, n_samples),
            'annual_inc': np.random.uniform(20000, 150000, n_samples),
            'dti': np.random.uniform(0, 40, n_samples),
            'fico_avg': np.random.uniform(600, 850, n_samples),
            'emp_length_numeric': np.random.uniform(0, 10, n_samples)
        })
        
        y = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        df = pd.concat([X, pd.Series(y, name='target')], axis=1)
        
        # Initialize and train models
        credit_model = CreditRiskModel()
        credit_model.prepare_data(df)
        credit_model.train_all_models()
        
        # Find best model
        best_model, best_model_name = credit_model.find_best_model()
        
        # Get model comparison
        comparison_df = credit_model.get_model_comparison()
        print("\n📊 Model Comparison:")
        print(comparison_df)
        
        # Save all models to the specified directory
        print("\n📝 Saving all models...")
        credit_model.save_all_models()
        
        # Save best model separately to the same directory
        output_dir = "D:\\Projek pribadi\\scholarship,exchange,pelatihan\\Rakamin-VIX-Intership-IDX\\models"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Created output directory: {output_dir}")
        
        filename = os.path.join(output_dir, f"best_{best_model_name}.pkl")
        print(f"📝 Saving best model ({best_model_name}) to: {filename}")
        
        success = credit_model.save_model(best_model, filename)
        
        if success:
            print("✅ Model saved successfully!")
        else:
            print("❌ Model saving failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()
