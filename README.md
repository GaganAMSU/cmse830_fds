Diabetes Analysis App

This project is an interactive Streamlit application for exploring, cleaning, analyzing, and modeling diabetes-related datasets. It integrates multiple real-world clinical datasets and supports user-uploaded or URL-based data. The app includes tools for missingness analysis, EDA, feature engineering, machine learning, diagnostics, parameter tuning, and model explainability.

Datasets Used
1. Scikit-Learn Diabetes Dataset

A built-in clinical dataset with ten standardized variables such as BMI, glucose, blood pressure, and age.
Source: scikit-learn.

2. Pima Indians Diabetes Dataset

A widely used medical dataset describing diabetes test results for Pima Indian women.
Source: Jason Brownlee (GitHub), originally from UCI.

3. NHANES Dataset

A curated subset of the National Health and Nutrition Examination Survey (NHANES), including BMI, glucose, cholesterol, and blood pressure.
Source: statOmics GitHub repository.

4. User-Uploaded or External CSV Data

Users can upload their own CSV files or load datasets from URLs, which are automatically integrated into the app.

Features
Data Cleaning & Missingness

Missingness summary tables, bar charts, and heatmaps

Automated options to drop or impute high-missing columns

Downloadable cleaned dataset

Exploratory Data Analysis

Summary statistics

Scatter plots, boxplots, correlation heatmaps

Automatic detection of numeric and categorical features

Feature Engineering

Standard scaling

Polynomial feature generation

One-hot encoding

Modeling

Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting

Performance metrics (R², RMSE, MAE)

Residual diagnostics and prediction error plots

Cross-validation and learning curves

Optional hyperparameter tuning through RandomizedSearchCV

SHAP model explainability when supported

Model export to .pkl

Documentation

Data dictionary generator

Combined dataset preview

Downloadable CSV export