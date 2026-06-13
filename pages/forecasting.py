import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import timedelta

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Expense Forecasting",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    finance_df = pd.read_csv(
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/datasets/finance_clean.csv"
    )

    finance_df["date"] = pd.to_datetime(
        finance_df["date"]
    )

    return finance_df


finance_df = load_data()

# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    xgb_model = joblib.load(
        "models/expense_xgb_model.pkl"
    )

    model_loaded = True

except:

    model_loaded = False

# ==========================================================
# HEADER
# ==========================================================

st.title("📈 Expense Forecasting Dashboard")

st.markdown(
    """
    Analyze historical spending trends and
    forecast future expenses using AI.
    """
)

st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================

avg_expense = round(

    finance_df[
        "monthly_expense_total"
    ].mean(),

    2

)

max_expense = round(

    finance_df[
        "monthly_expense_total"
    ].max(),

    2

)

min_expense = round(

    finance_df[
        "monthly_expense_total"
    ].min(),

    2

)

avg_income = round(

    finance_df[
        "monthly_income"
    ].mean(),

    2

)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Expense",
        f"${avg_expense:,.2f}"
    )

with col2:

    st.metric(
        "Maximum Expense",
        f"${max_expense:,.2f}"
    )

with col3:

    st.metric(
        "Minimum Expense",
        f"${min_expense:,.2f}"
    )

with col4:

    st.metric(
        "Average Income",
        f"${avg_income:,.2f}"
    )

# ==========================================================
# MONTHLY EXPENSE TREND
# ==========================================================

st.subheader(
    "Historical Expense Trend"
)

expense_ts = (

    finance_df

    .groupby(
        pd.Grouper(
            key="date",
            freq="M"
        )
    )

    ["monthly_expense_total"]

    .mean()

    .reset_index()

)

fig = px.line(

    expense_ts,

    x="date",

    y="monthly_expense_total",

    markers=True

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# YEARLY TREND
# ==========================================================

st.subheader(
    "Yearly Expense Analysis"
)

yearly_expense = (

    finance_df

    .groupby("year")

    ["monthly_expense_total"]

    .mean()

    .reset_index()

)

fig = px.bar(

    yearly_expense,

    x="year",

    y="monthly_expense_total",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# MONTHLY SEASONALITY
# ==========================================================

st.subheader(
    "Monthly Spending Pattern"
)

monthly_pattern = (

    finance_df

    .groupby("month")

    ["monthly_expense_total"]

    .mean()

    .reset_index()

)

fig = px.line(

    monthly_pattern,

    x="month",

    y="monthly_expense_total",

    markers=True

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# USER EXPENSE ANALYSIS
# ==========================================================

st.subheader(
    "User Expense Analysis"
)

selected_user = st.selectbox(

    "Select User",

    sorted(
        finance_df["user_id"].unique()
    )

)

user_data = finance_df[
    finance_df["user_id"] == selected_user
]

fig = px.line(

    user_data.sort_values("date"),

    x="date",

    y="monthly_expense_total",

    markers=True

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FORECAST SECTION
# ==========================================================

st.subheader(
    "12-Month Forecast Simulation"
)

latest_expense = (

    expense_ts

    ["monthly_expense_total"]

    .iloc[-1]

)

future_dates = pd.date_range(

    start=
    expense_ts["date"].max()
    + timedelta(days=30),

    periods=12,

    freq="M"

)

forecast_values = []

current_value = latest_expense

for i in range(12):

    growth_rate = 0.015

    current_value = (

        current_value *

        (1 + growth_rate)

    )

    forecast_values.append(
        current_value
    )

forecast_df = pd.DataFrame({

    "date": future_dates,

    "forecast": forecast_values

})

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=expense_ts["date"],

        y=expense_ts[
            "monthly_expense_total"
        ],

        mode="lines+markers",

        name="Historical"

    )

)

fig.add_trace(

    go.Scatter(

        x=forecast_df["date"],

        y=forecast_df["forecast"],

        mode="lines+markers",

        name="Forecast"

    )

)

fig.update_layout(

    title=
    "Expense Forecast"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FORECAST TABLE
# ==========================================================

st.subheader(
    "Forecasted Expenses"
)

forecast_display = forecast_df.copy()

forecast_display.columns = [

    "Forecast Date",

    "Predicted Expense"

]

st.dataframe(

    forecast_display,

    use_container_width=True

)

# ==========================================================
# EXPENSE DISTRIBUTION
# ==========================================================

st.subheader(
    "Expense Distribution"
)

fig = px.histogram(

    finance_df,

    x="monthly_expense_total",

    nbins=30

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# CATEGORY ANALYSIS
# ==========================================================

st.subheader(
    "Category Wise Expenses"
)

category_expense = (

    finance_df

    .groupby("category")

    ["monthly_expense_total"]

    .mean()

    .reset_index()

)

fig = px.bar(

    category_expense,

    x="category",

    y="monthly_expense_total"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# EXPENSE INSIGHTS
# ==========================================================

st.subheader(
    "AI Expense Insights"
)

if avg_expense > avg_income * 0.80:

    st.error(
        "Expenses consume a large portion of income."
    )

else:

    st.success(
        "Expense levels appear healthy."
    )

highest_month = monthly_pattern.loc[
    monthly_pattern[
        "monthly_expense_total"
    ].idxmax()
]

lowest_month = monthly_pattern.loc[
    monthly_pattern[
        "monthly_expense_total"
    ].idxmin()
]

st.info(

    f"Highest spending month: "
    f"{int(highest_month['month'])}"

)

st.info(

    f"Lowest spending month: "
    f"{int(lowest_month['month'])}"

)

# ==========================================================
# TOP SPENDERS
# ==========================================================

st.subheader(
    "Top Spending Users"
)

top_spenders = (

    finance_df

    .groupby("user_id")

    ["monthly_expense_total"]

    .mean()

    .reset_index()

    .sort_values(

        by="monthly_expense_total",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_spenders,

    use_container_width=True

)

# ==========================================================
# DOWNLOAD FORECAST
# ==========================================================

st.subheader(
    "Download Forecast Report"
)

csv = forecast_df.to_csv(
    index=False
)

st.download_button(

    label="Download Forecast",

    data=csv,

    file_name=
    "expense_forecast.csv",

    mime="text/csv"

)