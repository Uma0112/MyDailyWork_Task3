# MyDailyWork_Task3 
Customer Churn Prediction

Project Overview

This project aims to predict customer churn using machine learning models. The dataset contains customer information, including demographics, account details, and transaction history. The goal is to classify customers as churners or non-churners based on historical data.

  Files Included



Churn_Modelling.csv - The dataset used.



Customer_Churn_Prediction.ipynb - Jupyter Notebook with data preprocessing, model training, and evaluation.



best_gb_model.pkl, best_rf_model.pkl, best_lr_model.pkl - Trained models saved for deployment.

Dataset Details

File: Churn_Modelling.csv (from kaggle)

Target Variable: Exited (1 = Churned, 0 = Retained)

Features: Customer demographics, account balance, credit score, etc.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

SMOTE (for handling class imbalance)

Data Preprocessing

Removed unnecessary columns (RowNumber, CustomerId, Surname).

Encoded categorical variables (Geography, Gender).

Applied SMOTE to balance the dataset.

Standardized numerical features using StandardScaler.

Machine Learning Models

Model

Accuracy

ROC AUC Score

Gradient Boosting

82.05%

0.77

Random Forest

80.60%

0.76

Logistic Regression

73.30%

0.68

Gradient Boosting performed the best in terms of accuracy and AUC-ROC score.

Key Visualizations

Confusion Matrices

Feature Importance Analysis

ROC Curves for model comparison

Probability Distribution of Churners vs. Non-Churners

Usage Instructions

Clone the repository:

git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook to train and evaluate models.

Model Deployment

Trained models are saved as .pkl files:

best_gb_model.pkl

best_rf_model.pkl

best_lr_model.pkl

To load a model:

import pickle
with open("best_gb_model.pkl", "rb") as file:
    model = pickle.load(file)

Next Steps

Deploy the model using Streamlit.

Improve performance with hyperparameter tuning.

Explore deep learning methods for churn prediction.

For any queries, feel free to reach out!

