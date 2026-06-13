import os

if not os.path.exists(
    "models/isolation_forest_model.pkl"
):
    import download_models

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Personal Finance Advisor",
    page_icon="💰",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    finance = pd.read_csv(
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/datasets/finance_clean.csv"
    )

    paysim = pd.read_csv(
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/datasets/paysim_clean.csv"
    )

    return finance, paysim


finance_df, paysim_df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("💰 AI Personal Finance Advisor")
st.markdown(
    """
    Advanced Financial Intelligence Platform

    Financial Health • Goal Prediction • Budget Planning •
    Expense Forecasting • Fraud Detection • Anomaly Detection
    """
)

st.divider()

# ==========================================================
# KPI CALCULATIONS
# ==========================================================

total_users = finance_df["user_id"].nunique()

avg_health_score = round(
    finance_df["financial_health_score"].mean(),
    2
)

avg_risk_score = round(
    finance_df["financial_risk_score"].mean(),
    2
)

avg_goal_probability = round(
    finance_df["goal_success_probability"].mean(),
    2
)

fraud_count = int(
    paysim_df["isFraud"].sum()
)

anomaly_count = int(
    paysim_df["is_anomaly"].sum()
)

# ==========================================================
# KPI ROW
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Users",
        f"{total_users:,}"
    )

with col2:

    st.metric(
        "Avg Financial Health",
        avg_health_score
    )

with col3:

    st.metric(
        "Avg Risk Score",
        avg_risk_score
    )

col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Goal Success %",
        f"{avg_goal_probability:.1f}%"
    )

with col5:

    st.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

with col6:

    st.metric(
        "Anomalies Detected",
        f"{anomaly_count:,}"
    )

st.divider()

# ==========================================================
# FINANCIAL HEALTH DISTRIBUTION
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Financial Health Categories"
    )

    health_counts = (
        finance_df["health_category"]
        .value_counts()
        .reset_index()
    )

    fig = px.pie(
        health_counts,
        names="health_category",
        values="count",
        hole=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    st.subheader(
        "Financial Risk Categories"
    )

    risk_counts = (
        finance_df["financial_risk_category"]
        .value_counts()
        .reset_index()
    )

    fig = px.bar(
        risk_counts,
        x="financial_risk_category",
        y="count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# FINANCIAL PROFILES
# ==========================================================

st.subheader(
    "Financial Profile Distribution"
)

profile_counts = (
    finance_df["financial_profile"]
    .value_counts()
    .reset_index()
)

fig = px.bar(
    profile_counts,
    x="financial_profile",
    y="count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# HEALTH VS RISK
# ==========================================================

st.subheader(
    "Financial Health vs Risk Score"
)

fig = px.scatter(

    finance_df,

    x="financial_health_score",

    y="financial_risk_score",

    color="financial_profile",

    hover_data=[
        "user_id"
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# EXPENSE TREND
# ==========================================================

st.subheader(
    "Expense Trend Analysis"
)

finance_df["date"] = pd.to_datetime(
    finance_df["date"]
)

monthly_expense = (

    finance_df

    .groupby(
        pd.Grouper(
            key="date",
            freq="ME"
        )
    )

    ["monthly_expense_total"]

    .mean()

    .reset_index()

)

fig = px.line(

    monthly_expense,

    x="date",

    y="monthly_expense_total",

    markers=True

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FRAUD ANALYTICS
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Fraud Risk Distribution"
    )

    fraud_risk = (

        paysim_df["fraud_risk"]

        .value_counts()

        .reset_index()

    )

    fig = px.bar(

        fraud_risk,

        x="fraud_risk",

        y="count"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    st.subheader(
        "Anomaly Risk Distribution"
    )

    anomaly_risk = (

        paysim_df["anomaly_risk"]

        .value_counts()

        .reset_index()

    )

    fig = px.bar(

        anomaly_risk,

        x="anomaly_risk",

        y="count"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# TOP RISK USERS
# ==========================================================

st.subheader(
    "Top 10 Risk Users"
)

top_risk_users = (

    finance_df

    .sort_values(

        by="financial_risk_score",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_risk_users[

        [

            "user_id",

            "financial_health_score",

            "financial_risk_score",

            "goal_success_probability",

            "financial_profile"

        ]

    ],

    use_container_width=True

)

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.subheader(
    "System Status"
)

status_df = pd.DataFrame({

    "Module":[

        "Financial Health Engine",

        "Budget Recommendation",

        "Goal Prediction",

        "Expense Forecasting",

        "Fraud Detection",

        "Anomaly Detection",

        "Risk Scoring",

        "AI Advisor"

    ],

    "Status":[

        "Ready",

        "Ready",

        "Ready",

        "Ready",

        "Ready",

        "Ready",

        "Ready",

        "Ready"

    ]

})

st.dataframe(
    status_df,
    use_container_width=True
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "Navigation"
)

st.sidebar.success(
    """
    Use the Pages menu on the left
    to explore all modules.
    """
)

st.sidebar.info(
    f"""
    Users: {total_users:,}

    Finance Records: {len(finance_df):,}

    Transactions: {len(paysim_df):,}
    """
)