import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Financial Risk Dashboard",
    page_icon="⚠️",
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

st.title("⚠️ Financial Risk Dashboard")

st.markdown(
    """
    Comprehensive financial risk assessment system
    powered by financial health, debt burden,
    savings goals, fraud intelligence and anomaly analytics.
    """
)

st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================

avg_risk = round(

    finance_df[
        "financial_risk_score"
    ].mean(),

    2

)

max_risk = round(

    finance_df[
        "financial_risk_score"
    ].max(),

    2

)

min_risk = round(

    finance_df[
        "financial_risk_score"
    ].min(),

    2

)

high_risk_users = len(

    finance_df[

        finance_df[
            "financial_risk_category"
        ] == "High Risk"

    ]

)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Risk",
        avg_risk
    )

with col2:

    st.metric(
        "Highest Risk",
        max_risk
    )

with col3:

    st.metric(
        "Lowest Risk",
        min_risk
    )

with col4:

    st.metric(
        "High Risk Users",
        high_risk_users
    )

# ==========================================================
# USER SELECTION
# ==========================================================

st.subheader(
    "User Risk Analysis"
)

selected_user = st.selectbox(

    "Select User",

    sorted(
        finance_df["user_id"].unique()
    )

)

user_data = finance_df[
    finance_df["user_id"] == selected_user
].iloc[0]

# ==========================================================
# RISK SCORE GAUGE
# ==========================================================

risk_score = float(
    user_data[
        "financial_risk_score"
    ]
)

fig = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=risk_score,

        title={
            'text':
            "Financial Risk Score"
        },

        gauge={

            'axis':{
                'range':[0,100]
            },
            
            'bar': {
                    'color': 'lightblue'   # Moving gauge color
                },

            'steps':[

                {
                    'range':[0,30],
                    'color':'green'
                },

                {
                    'range':[30,60],
                    'color':'orange'
                },

                {
                    'range':[60,100],
                    'color':'red'
                }

            ]

        }

    )

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# USER RISK OVERVIEW
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Risk Category",
        user_data[
            "financial_risk_category"
        ]
    )

with col2:

    st.metric(
        "Health Score",
        round(
            user_data[
                "financial_health_score"
            ],
            2
        )
    )

with col3:

    st.metric(
        "Goal Success %",
        round(
            user_data[
                "goal_success_probability"
            ],
            2
        )
    )

# ==========================================================
# RISK COMPONENTS
# ==========================================================

st.subheader(
    "Risk Component Breakdown"
)

risk_components = pd.DataFrame({

    "Component":[

        "Credit Risk",

        "Debt Risk",

        "Goal Risk",

        "Health Risk",

        "Expense Risk",

        "Fraud Risk",

        "Anomaly Risk"

    ],

    "Score":[

        user_data[
            "credit_risk_score"
        ],

        user_data[
            "debt_risk_score"
        ],

        user_data[
            "goal_risk_score"
        ],

        user_data[
            "health_risk_score"
        ],

        user_data[
            "expense_risk_score"
        ],

        user_data[
            "fraud_risk_score"
        ],

        user_data[
            "anomaly_risk_score"
        ]

    ]

})

fig = px.bar(

    risk_components,

    x="Component",

    y="Score",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RISK CATEGORY DISTRIBUTION
# ==========================================================

st.subheader(
    "Risk Category Distribution"
)

risk_distribution = (

    finance_df

    ["financial_risk_category"]

    .value_counts()

    .reset_index()

)

fig = px.pie(

    risk_distribution,

    names="financial_risk_category",

    values="count",

    hole=0.5

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# HEALTH VS RISK
# ==========================================================

st.subheader(
    "Health Score vs Risk Score"
)

fig = px.scatter(

    finance_df,

    x="financial_health_score",

    y="financial_risk_score",

    color="financial_profile",

    hover_data=["user_id"]

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# PROFILE RISK ANALYSIS
# ==========================================================

st.subheader(
    "Financial Profile Risk Analysis"
)

profile_risk = (

    finance_df

    .groupby(
        "financial_profile"
    )

    ["financial_risk_score"]

    .mean()

    .reset_index()

)

fig = px.bar(

    profile_risk,

    x="financial_profile",

    y="financial_risk_score",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP RISK USERS
# ==========================================================

st.subheader(
    "Top 20 Risk Users"
)

top_risk_users = (

    finance_df

    .sort_values(

        by="financial_risk_score",

        ascending=False

    )

    .head(20)

)

st.dataframe(

    top_risk_users[

        [

            "user_id",

            "financial_risk_score",

            "financial_risk_category",

            "financial_health_score",

            "goal_success_probability",

            "financial_profile"

        ]

    ],

    use_container_width=True

)

# ==========================================================
# LOWEST RISK USERS
# ==========================================================

st.subheader(
    "Top 20 Lowest Risk Users"
)

low_risk_users = (

    finance_df

    .sort_values(

        by="financial_risk_score"

    )

    .head(20)

)

st.dataframe(

    low_risk_users[

        [

            "user_id",

            "financial_risk_score",

            "financial_risk_category",

            "financial_health_score",

            "goal_success_probability",

            "financial_profile"

        ]

    ],

    use_container_width=True

)

# ==========================================================
# RISK HISTOGRAM
# ==========================================================

st.subheader(
    "Risk Score Distribution"
)

fig = px.histogram(

    finance_df,

    x="financial_risk_score",

    nbins=30

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RISK INTELLIGENCE
# ==========================================================

st.subheader(
    "Risk Intelligence"
)

if risk_score > 60:

    st.error(
        "High financial risk detected."
    )

elif risk_score > 30:

    st.warning(
        "Moderate financial risk."
    )

else:

    st.success(
        "Low financial risk."
    )

if user_data["debt_to_income_ratio"] > 0.40:

    st.warning(
        "Debt burden exceeds recommended level."
    )

if user_data["expense_ratio"] > 0.80:

    st.warning(
        "Expense ratio is high."
    )

if user_data["goal_success_probability"] < 50:

    st.error(
        "Goal achievement probability is low."
    )

# ==========================================================
# AI INSIGHTS
# ==========================================================

st.subheader(
    "AI Financial Insights"
)

if (
    "ai_financial_insights"
    in finance_df.columns
):

    st.info(

        user_data[
            "ai_financial_insights"
        ]

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
# DOWNLOAD REPORT
# ==========================================================

st.subheader(
    "Download Risk Report"
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

    file_name=
    f"user_{selected_user}_risk_report.csv",

    mime="text/csv"

)