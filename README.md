# DecodeLabs-Project2-Fraud-Detection
Leak-Free Supervised Learning Fraud Detection Pipeline using SMOTE, Logistic Regression, and Random Forest.
## Project Overview
A leak-free Supervised Learning pipeline built in Python to detect fraudulent transactions on a highly imbalanced dataset (`99.45%` Legitimate vs. `0.55%` Fraudulent).

## Key Techniques
- **Zero-Leakage Architecture:** Applied a stratified 80/20 train-test split before any scaling or resampling.
- **Imbalanced Data Handling:** Integrated `SMOTE` inside `imblearn.pipeline.Pipeline` to generate synthetic minority samples strictly within cross-validation folds.
- **Models & Tuning:** Tuned **Logistic Regression** (with `StandardScaler`) and **Random Forest** using 5-fold `GridSearchCV` optimized for `ROC-AUC`.
- **Evaluation:** Evaluated on untouched test data using Confusion Matrices, Precision, Recall, F1-Score, and ROC-AUC.

## Test Set Results
- **Logistic Regression:** `0.9091` Recall | `0.9779` ROC-AUC
- **Random Forest:** `1.0000` Precision | `0.6364` Recall
