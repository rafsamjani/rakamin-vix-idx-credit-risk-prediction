#!/usr/bin/env python3
"""
Data Cleaning Script for Credit Risk Model
Creates cleaned dataset without SMOTE dependency
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Credit Risk Data Cleaning")
    print("="*40)

    # Load raw data
    print("Loading raw data...")
    try:
        df = pd.read_csv('../data/raw/loan_data_2007_2014.csv', low_memory=False)
        print(f"Raw data shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Raw dataset not found!")
        print("Expected location: ../data/raw/loan_data_2007_2014.csv")
        return

    # Data Cleaning Pipeline
    print("\nStarting data cleaning...")

    # 1. Filter for completed loans
    print("\n1. Filtering completed loans...")
    completed_loans = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(completed_loans)].copy()
    print(f"After filtering: {df.shape} loans")

    # 2. Clean key features
    print("\n2. Cleaning key features...")

    # Convert loan_status to binary
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
    print(f"Default rate: {df['loan_status'].mean()*100:.2f}%")

    # Clean and convert key numeric columns
    numeric_columns = {
        'int_rate': 'percentage',
        'dti': 'numeric',
        'annual_inc': 'numeric',
        'loan_amnt': 'numeric',
        'installment': 'numeric'
    }

    for col, col_type in numeric_columns.items():
        if col in df.columns:
            if col_type == 'percentage':
                df[col] = df[col].astype(str).str.replace('%', '').astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"  - {col}: {df[col].notna().sum()} non-null values")

    # Clean categorical columns
    categorical_columns = ['grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            print(f"  - {col}: {df[col].nunique()} unique values")

    # 3. Handle missing values
    print("\n3. Handling missing values...")

    # Fill missing numeric values with median
    for col in ['int_rate', 'dti', 'annual_inc', 'loan_amnt', 'installment']:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  - {col}: Filled {df[col].isna().sum()} missing with median ({median_val:.2f})")

    # Fill missing categorical with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f"  - {col}: Filled with '{mode_val}'")

    # 4. Remove extreme outliers
    print("\n4. Removing extreme outliers...")

    # Annual income > $1,000,000
    income_outliers = df['annual_inc'] > 1000000
    print(f"  - Income outliers removed: {income_outliers.sum()}")
    df = df[~income_outliers]

    # Loan amount > $100,000
    loan_outliers = df['loan_amnt'] > 100000
    print(f"  - Loan amount outliers removed: {loan_outliers.sum()}")
    df = df[~loan_outliers]

    # DTI > 100%
    dti_outliers = df['dti'] > 100
    print(f"  - DTI outliers removed: {dti_outliers.sum()}")
    df = df[~dti_outliers]

    # 5. Create final cleaned dataset
    print("\n5. Creating final cleaned dataset...")

    # Select essential columns for modeling
    essential_columns = [
        'loan_status', 'int_rate', 'dti', 'annual_inc', 'loan_amnt', 'installment',
        'grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose'
    ]

    # Ensure all essential columns exist
    available_columns = [col for col in essential_columns if col in df.columns]
    df_cleaned = df[available_columns].copy()

    print(f"Final cleaned dataset shape: {df_cleaned.shape}")
    print(f"Final default rate: {df_cleaned['loan_status'].mean()*100:.2f}%")

    # Save cleaned dataset
    output_path = '../data/processed/loan_data_cleaned.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

    # Display summary statistics
    print("\nDataset Summary:")
    print("="*40)
    print(f"Total loans: {len(df_cleaned):,}")
    print(f"Default rate: {df_cleaned['loan_status'].mean()*100:.2f}%")
    print(f"Average interest rate: {df_cleaned['int_rate'].mean():.1f}%")
    print(f"Average DTI: {df_cleaned['dti'].mean():.1f}%")
    print(f"Average loan amount: ${df_cleaned['loan_amnt'].mean():,.0f}")
    print(f"Average annual income: ${df_cleaned['annual_inc'].mean():,.0f}")

    print("\n✅ Data cleaning completed successfully!")

if __name__ == "__main__":
    main()