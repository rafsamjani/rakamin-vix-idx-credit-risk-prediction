#!/usr/bin/env python3
"""
Setup Script for Credit Risk Prediction Project Environment
=========================================================

This script creates and configures a virtual environment for the credit risk prediction project.

Author: Rafsamjani Anugrah
Date: 2024
"""

import subprocess
import sys
import os
from pathlib import Path

def create_virtual_environment():
    """Create virtual environment with project-specific name"""

    venv_name = "credit-risk-env"
    project_root = Path(__file__).parent
    venv_path = project_root / venv_name

    print("=" * 60)
    print("CREDIT RISK PREDICTION - ENVIRONMENT SETUP")
    print("=" * 60)
    print(f"Creating virtual environment: {venv_name}")
    print(f"Location: {venv_path}")

    try:
        # Create virtual environment
        print("\n[1/4] Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        print("✅ Virtual environment created successfully!")

        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip.exe"
            python_path = venv_path / "Scripts" / "python.exe"
        else:  # Linux/Mac
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"

        # Upgrade pip
        print("\n[2/4] Upgrading pip...")
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
        print("✅ Pip upgraded successfully!")

        # Install requirements
        requirements_path = project_root / "requirements.txt"
        if requirements_path.exists():
            print("\n[3/4] Installing project requirements...")
            subprocess.check_call([str(pip_path), "install", "-r", str(requirements_path)])
            print("✅ Requirements installed successfully!")
        else:
            print("\n⚠️  requirements.txt not found. Installing core packages...")
            core_packages = [
                "pandas==2.1.4",
                "numpy==1.24.3",
                "scikit-learn==1.3.2",
                "xgboost==2.0.3",
                "matplotlib==3.8.2",
                "seaborn==0.13.0",
                "plotly==5.17.0",
                "jupyter==1.0.0",
                "streamlit==1.29.0",
                "imbalanced-learn==0.12.2",
                "shap==0.42.1",
                "joblib==1.4.0"
            ]
            for package in core_packages:
                subprocess.check_call([str(pip_path), "install", package])
            print("✅ Core packages installed successfully!")

        # Install additional data science packages
        print("\n[4/4] Installing additional data science packages...")
        additional_packages = [
            "ipykernel>=6.29.0",
            "nbconvert>=7.16.0",
            "openpyxl>=3.1.2",
            "python-dotenv>=1.0.0",
            "tqdm>=4.66.1",
            "statsmodels>=0.14.0"
        ]

        for package in additional_packages:
            try:
                subprocess.check_call([str(pip_path), "install", package])
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {package}")

        print("\n" + "=" * 60)
        print("ENVIRONMENT SETUP COMPLETED!")
        print("=" * 60)

        # Activation instructions
        print(f"\n🎯 Virtual environment '{venv_name}' is ready!")
        print("\n📋 ACTIVATION INSTRUCTIONS:")

        if os.name == 'nt':  # Windows
            print(f"\n   Command Prompt:")
            print(f"   {venv_name}\\Scripts\\activate")
            print(f"\n   PowerShell:")
            print(f"   .\\{venv_name}\\Scripts\\Activate.ps1")
        else:  # Linux/Mac
            print(f"\n   Bash/Zsh:")
            print(f"   source {venv_name}/bin/activate")

        print(f"\n📁 PROJECT STRUCTURE:")
        print(f"   📁 notebooks/     - Jupyter notebooks for analysis")
        print(f"   📁 data/          - Data files")
        print(f"   📁 src/           - Python modules")
        print(f"   📁 docs/          - Documentation")
        print(f"   📁 dashboard/     - Streamlit dashboard")
        print(f"   📁 models/        - Trained models")

        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Activate the virtual environment")
        print(f"   2. Run: jupyter notebook")
        print(f"   3. Open: notebooks/credit_risk_analysis.ipynb")
        print(f"   4. Start with data exploration and cleaning")

        return str(venv_path)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during setup: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None

def create_project_notebooks():
    """Create placeholder notebooks for the analysis workflow"""

    notebooks_path = Path("notebooks")
    notebooks_path.mkdir(exist_ok=True)

    notebook_sequence = [
        ("01_data_exploration.ipynb", "Data Exploration and Understanding"),
        ("02_data_cleaning.ipynb", "Data Cleaning and Preprocessing"),
        ("03_feature_engineering.ipynb", "Feature Engineering and Selection"),
        ("04_model_development.ipynb", "Model Development and Training"),
        ("05_model_evaluation.ipynb", "Model Evaluation and Analysis"),
        ("06_business_insights.ipynb", "Business Insights and Recommendations"),
        ("credit_risk_analysis.ipynb", "Complete Analysis (Combined)")
    ]

    print(f"\n📝 Creating notebook templates...")

    for filename, title in notebook_sequence:
        notebook_path = notebooks_path / filename
        if not notebook_path.exists():
            # Create basic notebook structure
            notebook_content = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {title}\\n",
    "\\n",
    "**Author**: Rafsamjani Anugrah\\n",
    "**Date**: 2024\\n",
    "**Project**: Credit Risk Prediction - ID/X Partners\\n",
    "\\n",
    "## Overview\\n",
    "\\n",
    "This notebook covers {title.lower()} for the credit risk prediction project."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Import libraries\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "print("Libraries imported successfully!")"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''

            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(notebook_content)

        print(f"   ✅ {filename} - {title}")

    print(f"\\n📊 Notebook workflow structure created!")

if __name__ == "__main__":
    print("Credit Risk Prediction Project - Environment Setup")
    print("=" * 60)

    # Create virtual environment
    venv_path = create_virtual_environment()

    if venv_path:
        # Create notebook structure
        create_project_notebooks()

        print(f"\n🎉 SETUP COMPLETE!")
        print(f"\\nYour credit risk prediction environment is ready to use.")
        print(f"Start with data exploration in the notebooks folder.")
    else:
        print(f"\\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)