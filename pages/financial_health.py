import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Financial Health Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/datasets/finance_clean.csv"
    )

    return df


finance_df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("📊 Financial Health Dashboard")

st.markdown(
    """
    Analyze user financial well-being using
    Financial Health Score, Savings Performance,
    Credit Health, Investment Activity,
    and Financial Stability Metrics.
    """
)

st.divider()

# ==========================================================
# USER SELECTION
# ==========================================================

user_ids = sorted(
    finance_df["user_id"].unique()
)

selected_user = st.sidebar.selectbox(
    "Select User",
    user_ids
)

user_data = finance_df[
    finance_df["user_id"] == selected_user
].iloc[0]

# ==========================================================
# USER OVERVIEW
# ==========================================================

st.subheader("User Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Monthly Income",
        f"${user_data['monthly_income']:,.2f}"
    )

with col2:

    st.metric(
        "Monthly Expenses",
        f"${user_data['monthly_expense_total']:,.2f}"
    )

with col3:

    st.metric(
        "Actual Savings",
        f"${user_data['actual_savings']:,.2f}"
    )

with col4:

    st.metric(
        "Credit Score",
        int(user_data['credit_score'])
    )

st.divider()

# ==========================================================
# FINANCIAL HEALTH SCORE
# ==========================================================

st.subheader(
    "Financial Health Score"
)

health_score = float(
    user_data["financial_health_score"]
)

fig = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=health_score,

        title={
            'text':
            "Financial Health Score"
        },

        gauge={

            'axis': {
                'range': [0,100]
            },
            
            'bar': {
                    'color': 'lightblue'   # Moving gauge color
                },

            'steps': [

                {
                    'range':[0,40],
                    'color':'red'
                },

                {
                    'range':[40,70],
                    'color':'orange'
                },

                {
                    'range':[70,100],
                    'color':'green'
                }

            ]

        }

    )

)

st.plotly_chart(
    fig,
    use_container_width=True
)

health_category = user_data['health_category']

if health_category == "Good":

    st.success(
        f"Health Category: {health_category}"
    )

elif health_category == "Average":

    st.warning(
        f"Health Category: {health_category}"
    )

else:

    st.error(
        f"Health Category: {health_category}"
    )

# ==========================================================
# HEALTH COMPONENTS
# ==========================================================

st.subheader(
    "Health Score Components"
)

component_df = pd.DataFrame({

    "Metric":[

        "Savings Score",

        "Credit Score",

        "Debt Score",

        "Emergency Score",

        "Investment Score",

        "Expense Score"

    ],

    "Score":[

        user_data["savings_score"],

        user_data[
            "credit_score_normalized"
        ],

        user_data["debt_score"],

        user_data["emergency_score"],

        user_data["investment_score"],

        user_data["expense_score"]

    ]

})

fig = px.bar(

    component_df,

    x="Metric",

    y="Score",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FINANCIAL RATIOS
# ==========================================================

st.subheader(
    "Financial Ratios"
)

col1, col2 = st.columns(2)

with col1:

    ratio_df = pd.DataFrame({

        "Ratio":[

            "Expense Ratio",

            "Savings Ratio",

            "Investment Ratio",

            "Emergency Fund Ratio"

        ],

        "Value":[

            user_data["expense_ratio"],

            user_data["savings_ratio"],

            user_data["investment_ratio"],

            user_data["emergency_fund_ratio"]

        ]

    })

    fig = px.bar(

        ratio_df,

        x="Ratio",

        y="Value",

        text_auto=".2f"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(

        names=[

            "Essential Spending",

            "Discretionary Spending"

        ],

        values=[

            user_data["essential_spending"],

            user_data[
                "discretionary_spending"
            ]

        ],

        hole=0.5

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# FINANCIAL PROFILE
# ==========================================================

st.subheader(
    "Financial Profile"
)

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"Profile: {user_data['financial_profile']}"
    )

    st.write(
        f"Income Type: {user_data['income_type']}"
    )

    st.write(
        f"Cash Flow Status: {user_data['cash_flow_status']}"
    )

    st.write(
        f"Stress Level: {user_data['financial_stress_level']}"
    )

with col2:

    st.metric(
        "Financial Stability Index",
        round(
            user_data[
                "financial_stability_index"
            ],
            2
        )
    )

    st.metric(
        "Wealth Score",
        round(
            user_data[
                "wealth_score"
            ],
            2
        )
    )

# ==========================================================
# SAVINGS ANALYSIS
# ==========================================================

st.subheader(
    "Savings Analysis"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Savings Rate",
        round(
            user_data["savings_rate"],
            2
        )
    )

with col2:

    st.metric(
        "Actual Savings",
        round(
            user_data["actual_savings"],
            2
        )
    )

with col3:

    st.metric(
        "Budget Goal",
        round(
            user_data["budget_goal"],
            2
        )
    )

# ==========================================================
# CREDIT ANALYSIS
# ==========================================================

st.subheader(
    "Credit Analysis"
)

credit_score = int(
    user_data["credit_score"]
)

if credit_score >= 750:

    st.success(
        "Excellent Credit Score"
    )

elif credit_score >= 650:

    st.warning(
        "Good Credit Score"
    )

else:

    st.error(
        "Low Credit Score"
    )

# ==========================================================
# FINANCIAL HEALTH DISTRIBUTION
# ==========================================================

st.subheader(
    "Overall Health Distribution"
)

health_counts = (

    finance_df

    ["health_category"]

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

# ==========================================================
# TOP HEALTHY USERS
# ==========================================================

st.subheader(
    "Top Financially Healthy Users"
)

top_users = (

    finance_df

    .sort_values(

        by="financial_health_score",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_users[

        [

            "user_id",

            "financial_health_score",

            "health_category",

            "monthly_income",

            "actual_savings"

        ]

    ],

    use_container_width=True

)

# ==========================================================
# AI INSIGHTS
# ==========================================================

st.subheader(
    "AI Financial Insights"
)

if "ai_financial_insights" in finance_df.columns:

    st.info(
        user_data[
            "ai_financial_insights"
        ]
    )

if "financial_recommendations" in finance_df.columns:

    st.success(
        user_data[
            "financial_recommendations"
        ]
    )

if "financial_alerts" in finance_df.columns:

    alerts = str(
        user_data["financial_alerts"]
    )

    if alerts.strip():

        st.error(alerts)

    else:

        st.success(
            "No Financial Alerts"
        )

# ==========================================================
# DOWNLOAD USER REPORT
# ==========================================================

st.subheader(
    "Download User Report"
)

report = pd.DataFrame(
    [user_data]
)

csv = report.to_csv(
    index=False
)

st.download_button(

    label="Download Report",

    data=csv,

    file_name=f"user_{selected_user}_financial_report.csv",

    mime="text/csv"

)