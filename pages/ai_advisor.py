import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/datasets/finance_clean.csv"
    )


finance_df = load_data()

# ==========================================================
# HEADER
# ==========================================================

st.title("🤖 AI Financial Advisor")

st.markdown(
    """
    Personalized AI-powered financial guidance platform
    combining financial health, goal planning,
    risk assessment, budgeting and intelligent recommendations.
    """
)

st.divider()

# ==========================================================
# USER SELECTION
# ==========================================================

selected_user = st.sidebar.selectbox(

    "Select User",

    sorted(
        finance_df["user_id"].unique()
    )

)

user_data = finance_df[
    finance_df["user_id"] == selected_user
].iloc[0]

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.subheader(
    "Executive Financial Summary"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Health Score",
        round(
            user_data[
                "financial_health_score"
            ],
            2
        )
    )

with col2:

    st.metric(
        "Risk Score",
        round(
            user_data[
                "financial_risk_score"
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

with col4:

    st.metric(
        "Credit Score",
        int(
            user_data[
                "credit_score"
            ]
        )
    )

# ==========================================================
# HEALTH & RISK GAUGES
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=float(
                user_data[
                    "financial_health_score"
                ]
            ),

            title={
                "text":
                "Financial Health"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },
                
                'bar': {
                    'color': 'lightblue'   # Moving gauge color
                },

                "steps":[

                    {
                        "range":[0,40],
                        "color":"red"
                    },

                    {
                        "range":[40,70],
                        "color":"orange"
                    },

                    {
                        "range":[70,100],
                        "color":"green"
                    }

                ]

            }

        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=float(
                user_data[
                    "financial_risk_score"
                ]
            ),

            title={
                "text":
                "Financial Risk"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },
                
                'bar': {
                    'color': 'lightblue'   # Moving gauge color
                },

                "steps":[

                    {
                        "range":[0,30],
                        "color":"green"
                    },

                    {
                        "range":[30,60],
                        "color":"orange"
                    },

                    {
                        "range":[60,100],
                        "color":"red"
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
# FINANCIAL PROFILE
# ==========================================================

st.subheader(
    "Financial Profile"
)

profile_df = pd.DataFrame({

    "Metric":[

        "Income",

        "Expenses",

        "Savings",

        "Investments",

        "Emergency Fund"

    ],

    "Amount":[

        user_data[
            "monthly_income"
        ],

        user_data[
            "monthly_expense_total"
        ],

        user_data[
            "actual_savings"
        ],

        user_data[
            "investment_amount"
        ],

        user_data[
            "emergency_fund"
        ]

    ]

})

fig = px.bar(

    profile_df,

    x="Metric",

    y="Amount",

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

ratio_df = pd.DataFrame({

    "Ratio":[

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

    x="Ratio",

    y="Value",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# GOAL ANALYSIS
# ==========================================================

st.subheader(
    "Goal Achievement Analysis"
)

goal_probability = float(
    user_data[
        "goal_success_probability"
    ]
)

if goal_probability >= 80:

    st.success(
        f"Goal Success Probability: {goal_probability:.2f}%"
    )

elif goal_probability >= 50:

    st.warning(
        f"Goal Success Probability: {goal_probability:.2f}%"
    )

else:

    st.error(
        f"Goal Success Probability: {goal_probability:.2f}%"
    )

# ==========================================================
# RISK BREAKDOWN
# ==========================================================

st.subheader(
    "Risk Breakdown"
)

risk_df = pd.DataFrame({

    "Risk Factor":[

        "Credit",

        "Debt",

        "Goal",

        "Health",

        "Expense",

        "Fraud",

        "Anomaly"

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

    risk_df,

    x="Risk Factor",

    y="Score",

    text_auto=".2f"

)

st.plotly_chart(
    fig,
    use_container_width=True
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

# ==========================================================
# RECOMMENDATIONS
# ==========================================================

st.subheader(
    "Personalized Recommendations"
)

if (
    "financial_recommendations"
    in finance_df.columns
):

    recommendations = str(

        user_data[
            "financial_recommendations"
        ]

    ).split("|")

    for rec in recommendations:

        if rec.strip():

            st.success(
                rec.strip()
            )

# ==========================================================
# ALERTS
# ==========================================================

st.subheader(
    "Financial Alerts"
)

alerts = str(
    user_data[
        "financial_alerts"
    ]
)

if alerts.strip():

    alert_list = alerts.split("|")

    for alert in alert_list:

        if alert.strip():

            st.error(
                alert.strip()
            )

else:

    st.success(
        "No active financial alerts."
    )

# ==========================================================
# USER RANKING
# ==========================================================

st.subheader(
    "User Ranking"
)

finance_df[
    "health_rank"
] = (

    finance_df[
        "financial_health_score"
    ]

    .rank(
        ascending=False
    )

)

user_rank = int(

    finance_df.loc[

        finance_df[
            "user_id"
        ] == selected_user,

        "health_rank"

    ].iloc[0]

)

st.metric(
    "Health Rank",
    f"#{user_rank}"
)

# ==========================================================
# TOP HEALTHY USERS
# ==========================================================

st.subheader(
    "Top Healthy Users"
)

top_users = (

    finance_df

    .sort_values(

        by=
        "financial_health_score",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_users[

        [

            "user_id",

            "financial_health_score",

            "financial_risk_score",

            "goal_success_probability"

        ]

    ],

    use_container_width=True

)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.subheader(
    "Download Advisor Report"
)

report = pd.DataFrame(
    [user_data]
)

csv = report.to_csv(
    index=False
)

st.download_button(

    label="Download Financial Report",

    data=csv,

    file_name=
    f"user_{selected_user}_advisor_report.csv",

    mime="text/csv"

)