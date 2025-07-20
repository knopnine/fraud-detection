# Fraud Detection Project (In Progress)
This project is an exploration of building a machine learning model to detect fraudulent transactions. The main goal is to compare different classification models and techniques for handling imbalanced data, with all experiments tracked using MLflow.

## Current Status
- Data Analysis: The fraud_detection.csv dataset has been loaded and analyzed to understand its features and imbalanced nature.
- Preprocessing: A pipeline has been created to scale numerical data and one-hot encode categorical data.
- Modeling:
    - Several baseline models have been trained, including Logistic Regression, RandomForest, and XGBoost.
    - Experiments are being run with various imbalance handling techniques like class weighting, SMOTE-Tomek, and ADASYN.
    - Hyperparameter tuning is underway to optimize the best-performing models.
- Experiment Tracking: All model runs, parameters, and metrics are logged to an MLflow server for comparison.