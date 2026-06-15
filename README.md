# 💰 AI Personal Finance Advisor

## 🚀 Overview

AI Personal Finance Advisor is an advanced end-to-end Data Science and Machine Learning project designed to help individuals analyze their financial health, optimize spending habits, predict future expenses, detect fraudulent transactions, identify anomalies, and generate personalized financial recommendations.

The system combines multiple Machine Learning models, Financial Analytics, Fraud Detection, Anomaly Detection, Risk Scoring, Forecasting, and AI-powered insights into a single intelligent financial platform.

Built using Python, XGBoost, Isolation Forest, Financial Analytics, and Streamlit, this project delivers real-time financial intelligence and decision support.

---

## 🎯 Problem Statement

Managing personal finances effectively is challenging due to:

- Lack of visibility into spending habits
- Poor budgeting strategies
- Uncertainty in achieving financial goals
- Fraudulent financial activities
- Hidden anomalies in transactions
- Inability to accurately measure financial health

This project addresses these challenges using Data Science and Machine Learning by providing intelligent financial analytics and recommendations.

---
# Demo: https://ai-finance-advisor-kswjftenxxn6ewwqg5xa8w.streamlit.app/



https://github.com/user-attachments/assets/5f58edf5-aefd-4156-ab0b-520b3334573c





---

## 🎯 Project Objectives

- Analyze personal financial behavior
- Measure financial health using custom scoring
- Recommend personalized budgets
- Predict savings goal achievement
- Forecast future expenses
- Detect fraudulent transactions
- Identify anomalous financial activities
- Calculate financial risk scores
- Generate AI-powered financial insights
- Build an interactive financial advisor dashboard

---

# 📊 Datasets Used

## 1. PaySim Financial Transaction Dataset

### Description

PaySim is a synthetic mobile money transaction dataset that simulates real-world financial transactions.

### Dataset Size

- 6.3 Million Transactions
- Highly Imbalanced Dataset
- Suitable for Fraud Detection and Anomaly Detection

### Features

| Feature | Description |
|----------|-------------|
| step | Time step |
| type | Transaction type |
| amount | Transaction amount |
| oldbalanceOrg | Sender balance before transaction |
| newbalanceOrig | Sender balance after transaction |
| oldbalanceDest | Receiver balance before transaction |
| newbalanceDest | Receiver balance after transaction |
| isFraud | Fraud label |
| isFlaggedFraud | Fraud flag |

### Applications

- Fraud Detection
- Anomaly Detection
- Financial Risk Analysis

---

## 2. Personal Finance Tracker Dataset

### Description

A comprehensive dataset containing personal financial information including income, expenses, savings, investments, debt ratios, and financial behavior indicators.

### Features

- monthly_income
- monthly_expense_total
- savings_rate
- credit_score
- debt_to_income_ratio
- investment_amount
- emergency_fund
- actual_savings
- financial_stress_level
- financial_advice_score
- transaction_count
- savings_goal_met

### Applications

- Financial Health Analysis
- Budget Recommendation
- Goal Prediction
- Risk Scoring
- AI Financial Advisor

---

# 🏗️ System Architecture

```text
Raw Datasets
      ↓
Data Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Financial Health Engine
      ↓
Budget Recommendation Engine
      ↓
Goal Prediction Model
      ↓
Expense Forecasting Model
      ↓
Fraud Detection Model
      ↓
Anomaly Detection Model
      ↓
Financial Risk Scoring Engine
      ↓
AI Insight Generator
      ↓
Streamlit Dashboard
```

---

# ⚙️ Technologies Used

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Visualization

- Plotly
- Matplotlib
- Seaborn

## Machine Learning

- Scikit-Learn
- XGBoost
- Isolation Forest

## Deployment

- Streamlit

## Model Persistence

- Joblib

---

# 🔍 Data Preprocessing

## Personal Finance Dataset

### Operations Performed

- Missing Value Handling
- Duplicate Removal
- Datetime Conversion
- Feature Engineering
- Outlier Analysis
- Data Validation

### Engineered Features

- Expense Ratio
- Savings Ratio
- Investment Ratio
- Emergency Fund Ratio
- Wealth Score
- Financial Stability Index
- Financial Health Score

---

## PaySim Dataset

### Operations Performed

- Fraud Data Analysis
- Transaction Feature Engineering
- Log Transformations
- Balance Difference Features
- Outlier Detection

### Engineered Features

- amount_log
- orig_balance_diff
- dest_balance_diff
- high_amount_flag

---

# 📈 Exploratory Data Analysis (EDA)

Performed comprehensive exploratory analysis including:

- Financial Health Distribution
- Income Analysis
- Expense Analysis
- Savings Analysis
- Financial Profile Analysis
- Risk Analysis
- Fraud Pattern Analysis
- Anomaly Analysis
- Correlation Analysis

---

# 💚 Financial Health Score Engine

A custom scoring system developed to measure overall financial wellness.

### Components

- Savings Score
- Credit Score
- Debt Score
- Emergency Fund Score
- Investment Score
- Expense Score

### Output

**Financial Health Score (0–100)**

| Score Range | Category |
|------------|----------|
| 80 - 100 | Excellent |
| 60 - 79 | Good |
| 40 - 59 | Average |
| 0 - 39 | Poor |

---

# 💸 Budget Recommendation Engine

Automatically recommends optimal budget allocation based on financial profile.

### Recommended Allocation

- Essentials
- Savings
- Investments
- Lifestyle

### Financial Profiles

- Saver
- Balanced
- Investor
- Conservative

---

# 🎯 Goal-Based Financial Planning

Predicts whether users will achieve their financial goals.

### Model

**Random Forest Classifier**

### Input Features

- Income
- Expenses
- Savings Rate
- Investments
- Credit Score
- Debt Ratio

### Outputs

- Goal Success Probability
- Goal Risk Category

---

# 📊 Expense Forecasting

Predicts future monthly expenses.

### Model

**XGBoost Regressor**

### Features

- Historical Expenses
- Income Trends
- Spending Behavior
- Financial Scenario

### Outputs

- Predicted Future Expenses
- Spending Trends

---

# 🛡️ Fraud Detection System

Detects fraudulent transactions using supervised machine learning.

### Model

**XGBoost Classifier**

### Features

- Transaction Amount
- Balance Differences
- Transaction Type
- High Amount Flag

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

### Outputs

- Fraud Probability
- Fraud Risk Category

---

# 🚨 Anomaly Detection System

Identifies unusual transaction behavior using unsupervised learning.

### Model

**Isolation Forest**

### Features

- Amount
- Balance Differences
- Transaction Patterns

### Outputs

- Anomaly Score
- Anomaly Risk Category

### Additional Validation Layer

Business Rule Validation:

- Sender Balance Verification
- Receiver Balance Verification
- Balance Mismatch Detection

---

# ⚠️ Financial Risk Scoring System

Combines multiple risk indicators into a unified risk score.

### Components

- Credit Risk
- Debt Risk
- Goal Risk
- Health Risk
- Expense Risk
- Fraud Risk
- Anomaly Risk

### Output

**Financial Risk Score (0–100)**

| Score Range | Category |
|------------|----------|
| 0 - 30 | Low Risk |
| 31 - 70 | Medium Risk |
| 71 - 100 | High Risk |

---

# 🤖 AI Financial Insight Generator

Generates personalized recommendations based on user financial behavior.

### Example Insights

- Increase monthly savings
- Reduce discretionary spending
- Improve debt repayment strategy
- Build emergency funds
- Increase investment allocation
- Improve financial discipline

---

# 📊 Streamlit Dashboard Modules

## 🏠 Dashboard Home

- Executive Summary
- KPI Overview
- Financial Analytics

## 💚 Financial Health

- Health Score Analysis
- Health Category Visualization

## 💸 Budget Recommendation

- Personalized Budget Allocation
- Financial Profile Analysis

## 🎯 Goal Prediction

- Goal Achievement Probability
- Goal Risk Analysis

## 📈 Expense Forecasting

- Future Expense Prediction
- Trend Analysis

## 🛡️ Fraud Detection

- Real-Time Fraud Prediction
- Fraud Analytics

## 🚨 Anomaly Detection

- Real-Time Anomaly Detection
- Anomaly Analytics

## ⚠️ Financial Risk

- Risk Score Dashboard
- Risk Intelligence

## 🤖 AI Financial Advisor

- Personalized Recommendations
- Financial Alerts
- AI Insights

---

# 📁 Project Structure

```text
AI-Finance-Advisor/
│
├── app.py
│
├── pages/
│   ├── financial_health.py
│   ├── budget_recommendation.py
│   ├── goal_prediction.py
│   ├── expense_forecasting.py
│   ├── fraud_detection.py
│   ├── anomaly_detection.py
│   ├── financial_risk.py
│   └── ai_advisor.py
│
├── datasets/
│   ├── finance_clean.csv
│   └── paysim_clean.csv
│
├── models/
│   ├── goal_prediction_model.pkl
│   ├── expense_xgb_model.pkl
│   ├── fraud_xgb_model.pkl
│   └── isolation_forest_model.pkl
│
├── scalers/
│   ├── anomaly_scaler.pkl
│   └── goal_scaler.pkl
│
├── utils/
│   └── load_models.py
│
├── download_models.py
│
├── requirements.txt
│
└── README.md
```

---

# 📦 Installation

### Clone Repository

```bash
git clone https://github.com/kushvantho15

https://github.com/user-attachments/assets/05525274-417f-4853-85ee-d99df75e4a76





/AI-Finance-Advisor.git
```

### Navigate to Project Folder

```bash
cd AI-Finance-Advisor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

# 🌟 Key Highlights

✅ End-to-End Data Science Project

✅ Financial Health Analytics

✅ Budget Recommendation System

✅ Goal Achievement Prediction

✅ Expense Forecasting using XGBoost

✅ Fraud Detection using XGBoost

✅ Anomaly Detection using Isolation Forest

✅ Financial Risk Intelligence

✅ AI-Based Personalized Recommendations

✅ Interactive Streamlit Dashboard

✅ Real-Time Financial Analytics

---

# 📈 Machine Learning Models

| Module | Algorithm |
|----------|------------|
| Goal Prediction | Random Forest Classifier |
| Expense Forecasting | XGBoost Regressor |
| Fraud Detection | XGBoost Classifier |
| Anomaly Detection | Isolation Forest |

---

# 🔮 Future Enhancements

- LLM-Powered Financial Assistant
- Real-Time Banking API Integration
- OCR-Based Receipt Analysis
- Investment Portfolio Optimization
- Cryptocurrency Risk Analytics
- Voice-Based Financial Advisor
- Multi-Language Support
- Real-Time Transaction Monitoring
- RAG-Based Financial Knowledge Assistant
- AI-Powered Financial Planning Agent

---

# 👨‍💻 Author

## Kushvanth Venkata Karthik

**Aspiring Data Scientist | Machine Learning Engineer | Full Stack Developer**

Passionate about building AI-powered solutions that solve real-world problems using Data Science, Machine Learning, and Generative AI.

### Connect With Me

- LinkedIn: (https://www.linkedin.com/in/kushvanth-badisa/)
- Email: badisakushvanthvenkatakarthik@gmail.com

---

## ⭐ If you found this project useful, don't forget to star the repository!
