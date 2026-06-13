import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Budget Recommendation Engine",
    page_icon="💰",
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

st.title("💰 Budget Recommendation Engine")

st.markdown(
    """
    AI-powered budgeting system that analyzes
    financial behavior and recommends an
    optimized monthly budget allocation.
    """
)

st.divider()

# ==========================================================
# USER SELECTION
# ==========================================================

selected_user = st.sidebar.selectbox(
    "Select User",
    sorted(finance_df["user_id"].unique())
)

user_data = finance_df[
    finance_df["user_id"] == selected_user
].iloc[0]

# ==========================================================
# USER OVERVIEW
# ==========================================================

st.subheader("User Financial Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Monthly Income",
        f"${user_data['monthly_income']:,.2f}"
    )

with col2:

    st.metric(
        "Monthly Expense",
        f"${user_data['monthly_expense_total']:,.2f}"
    )

with col3:

    st.metric(
        "Actual Savings",
        f"${user_data['actual_savings']:,.2f}"
    )

with col4:

    st.metric(
        "Financial Health",
        round(
            user_data[
                'financial_health_score'
            ],
            2
        )
    )

st.divider()

# ==========================================================
# FINANCIAL PROFILE
# ==========================================================

st.subheader("Financial Profile")

profile = user_data["financial_profile"]

if profile == "Saver":

    st.success(
        f"Profile: {profile}"
    )

elif profile == "Balanced":

    st.info(
        f"Profile: {profile}"
    )

elif profile == "Investor":

    st.warning(
        f"Profile: {profile}"
    )

else:

    st.error(
        f"Profile: {profile}"
    )

# ==========================================================
# BUDGET CLUSTER INFO
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Budget Cluster",
        int(
            user_data[
                "budget_cluster"
            ]
        )
    )

with col2:

    st.metric(
        "Risk Category",
        user_data[
            "financial_risk_category"
        ]
    )

# ==========================================================
# RECOMMENDED BUDGET
# ==========================================================

st.subheader(
    "Recommended Budget Allocation"
)

budget_data = user_data[
    "recommended_budget"
]

if isinstance(
    budget_data,
    str
):

    budget_data = ast.literal_eval(
        budget_data
    )

budget_df = pd.DataFrame({

    "Category":
    list(
        budget_data.keys()
    ),

    "Amount":
    list(
        budget_data.values()
    )

})

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(

        budget_df,

        names="Category",

        values="Amount",

        hole=0.5

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.bar(

        budget_df,

        x="Category",

        y="Amount",

        text_auto=".2f"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# ACTUAL VS RECOMMENDED
# ==========================================================

st.subheader(
    "Actual vs Recommended Allocation"
)

actual_df = pd.DataFrame({

    "Category":[

        "Essential Spending",

        "Discretionary Spending",

        "Savings",

        "Investments"

    ],

    "Actual":[

        user_data[
            "essential_spending"
        ],

        user_data[
            "discretionary_spending"
        ],

        user_data[
            "actual_savings"
        ],

        user_data[
            "investment_amount"
        ]

    ]

})

fig = px.bar(

    actual_df,

    x="Category",

    y="Actual",

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
    "Budget Performance Ratios"
)

ratio_df = pd.DataFrame({

    "Metric":[

        "Expense Ratio",

        "Savings Ratio",

        "Investment Ratio",

        "Emergency Fund Ratio"

    ],

    "Value":[

        user_data[
            "expense_ratio"
        ],

        user_data[
            "savings_ratio"
        ],

        user_data[
            "investment_ratio"
        ],

        user_data[
            "emergency_fund_ratio"
        ]

    ]

})

fig = px.bar(

    ratio_df,

    x="Metric",

    y="Value",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# PROFILE DISTRIBUTION
# ==========================================================

st.subheader(
    "Financial Profile Distribution"
)

profile_counts = (

    finance_df

    ["financial_profile"]

    .value_counts()

    .reset_index()

)

fig = px.pie(

    profile_counts,

    names="financial_profile",

    values="count",

    hole=0.5

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# PROFILE COMPARISON
# ==========================================================

st.subheader(
    "Profile Performance Comparison"
)

profile_stats = (

    finance_df

    .groupby(
        "financial_profile"
    )

    [

        [

            "financial_health_score",

            "actual_savings",

            "monthly_income"

        ]

    ]

    .mean()

    .reset_index()

)

fig = px.bar(

    profile_stats,

    x="financial_profile",

    y="financial_health_score",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# AI RECOMMENDATIONS
# ==========================================================

st.subheader(
    "Budget Recommendations"
)

if (
    "financial_recommendations"
    in finance_df.columns
):

    st.success(

        user_data[
            "financial_recommendations"
        ]

    )

# ==========================================================
# AI INSIGHTS
# ==========================================================

st.subheader(
    "Budget Insights"
)

insights = []

if user_data["expense_ratio"] > 0.80:

    insights.append(
        "Expenses consume more than 80% of income."
    )

if user_data["savings_ratio"] < 0.15:

    insights.append(
        "Savings ratio is below recommended levels."
    )

if user_data["investment_ratio"] < 0.10:

    insights.append(
        "Investment allocation can be improved."
    )

if user_data["emergency_fund_ratio"] < 0.15:

    insights.append(
        "Emergency fund reserve is relatively low."
    )

if len(insights) == 0:

    insights.append(
        "Current budget allocation appears healthy."
    )

for insight in insights:

    st.info(insight)

# ==========================================================
# TOP SAVERS
# ==========================================================

st.subheader(
    "Top Savers"
)

top_savers = (

    finance_df

    .sort_values(

        by="actual_savings",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_savers[

        [

            "user_id",

            "financial_profile",

            "actual_savings",

            "financial_health_score"

        ]

    ],

    use_container_width=True

)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.subheader(
    "Download Budget Report"
)

report = pd.DataFrame(
    [user_data]
)

csv = report.to_csv(
    index=False
)

st.download_button(

    label="Download Budget Report",

    data=csv,

    file_name=f"user_{selected_user}_budget_report.csv",

    mime="text/csv"

)