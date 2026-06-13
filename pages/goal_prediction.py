import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Goal Prediction",
    page_icon="🎯",
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

    return finance_df


finance_df = load_data()

# ==========================================================
# LOAD MODEL
# ==========================================================

goal_model = joblib.load(
    "models/goal_prediction_model.pkl"
)

goal_scaler = joblib.load(
    "scalers/goal_scaler.pkl"
)

# ==========================================================
# HEADER
# ==========================================================

st.title("🎯 Goal Achievement Prediction")

st.markdown(
    """
    Predict the probability of achieving
    a financial savings goal using AI.
    """
)

st.divider()

# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader(
    "Enter Financial Information"
)

col1, col2 = st.columns(2)

with col1:

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0
    )

    monthly_expense_total = st.number_input(
        "Monthly Expense",
        min_value=0.0,
        value=3500.0
    )

    savings_rate = st.slider(
        "Savings Rate",
        0.0,
        1.0,
        0.20
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )

    debt_to_income_ratio = st.slider(
        "Debt-To-Income Ratio",
        0.0,
        1.0,
        0.30
    )

    loan_payment = st.number_input(
        "Loan Payment",
        min_value=0.0,
        value=500.0
    )

with col2:

    investment_amount = st.number_input(
        "Investment Amount",
        min_value=0.0,
        value=300.0
    )

    emergency_fund = st.number_input(
        "Emergency Fund",
        min_value=0.0,
        value=1000.0
    )

    transaction_count = st.number_input(
        "Transaction Count",
        min_value=0,
        value=50
    )

    financial_advice_score = st.slider(
        "Financial Advice Score",
        0.0,
        100.0,
        50.0
    )

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

expense_ratio = (
    monthly_expense_total /
    monthly_income
    if monthly_income > 0
    else 0
)

savings_ratio = max(
    0,
    (
        monthly_income -
        monthly_expense_total
    ) / monthly_income
) if monthly_income > 0 else 0

investment_ratio = (
    investment_amount /
    monthly_income
) if monthly_income > 0 else 0

# ==========================================================
# APPROXIMATE HEALTH SCORE
# ==========================================================

financial_health_score = (

    savings_ratio * 40

    +

    (credit_score / 850) * 30

    +

    (1 - debt_to_income_ratio) * 30

)

financial_health_score = min(
    100,
    financial_health_score
)

# ==========================================================
# PREDICTION
# ==========================================================

if st.button(
    "Predict Goal Achievement"
):

    input_data = pd.DataFrame({

        'monthly_income':
        [monthly_income],

        'monthly_expense_total':
        [monthly_expense_total],

        'savings_rate':
        [savings_rate],

        'credit_score':
        [credit_score],

        'debt_to_income_ratio':
        [debt_to_income_ratio],

        'loan_payment':
        [loan_payment],

        'investment_amount':
        [investment_amount],

        'emergency_fund':
        [emergency_fund],

        'transaction_count':
        [transaction_count],

        'financial_advice_score':
        [financial_advice_score],

        'expense_ratio':
        [expense_ratio],

        'savings_ratio':
        [savings_ratio],

        'investment_ratio':
        [investment_ratio],

        'financial_health_score':
        [financial_health_score]

    })

    # ======================================================
    # SCALE
    # ======================================================

    scaled_input = goal_scaler.transform(
        input_data
    )

    prediction = goal_model.predict(
        scaled_input
    )[0]

    probability = (

        goal_model.predict_proba(
            scaled_input
        )[0][1]

        * 100

    )

    # ======================================================
    # RESULTS
    # ======================================================

    st.divider()

    st.subheader(
        "Prediction Results"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Success Probability",
            f"{probability:.2f}%"
        )

    with col2:

        st.metric(
            "Financial Health",
            f"{financial_health_score:.2f}"
        )

    with col3:

        if prediction == 1:

            st.success(
                "Goal Achievable"
            )

        else:

            st.error(
                "Goal At Risk"
            )

    # ======================================================
    # GAUGE CHART
    # ======================================================

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability,

            title={
                'text':
                "Goal Success Probability"
            },

            gauge={

                'axis': {
                    'range':[0,100]
                },
                
                'bar': {
                    'color': 'lightblue'   # Moving gauge color
                },

                'steps':[

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

    # ======================================================
    # GOAL RISK
    # ======================================================

    st.subheader(
        "Goal Risk Assessment"
    )

    if probability >= 80:

        st.success(
            "Low Risk"
        )

    elif probability >= 60:

        st.warning(
            "Moderate Risk"
        )

    else:

        st.error(
            "High Risk"
        )

    # ======================================================
    # AI RECOMMENDATIONS
    # ======================================================

    st.subheader(
        "AI Recommendations"
    )

    recommendations = []

    if savings_ratio < 0.15:

        recommendations.append(
            "Increase monthly savings."
        )

    if debt_to_income_ratio > 0.35:

        recommendations.append(
            "Reduce debt obligations."
        )

    if investment_ratio < 0.10:

        recommendations.append(
            "Increase investment allocation."
        )

    if emergency_fund < (
        monthly_expense_total * 3
    ):

        recommendations.append(
            "Build a stronger emergency fund."
        )

    if probability > 80:

        recommendations.append(
            "You are on track to achieve your goal."
        )

    for rec in recommendations:

        st.info(rec)

# ==========================================================
# EXISTING USER ANALYSIS
# ==========================================================

st.divider()

st.subheader(
    "Existing User Goal Analytics"
)

goal_distribution = (

    finance_df

    ["goal_risk"]

    .value_counts()

    .reset_index()
)

fig = px.pie(

    goal_distribution,

    names="goal_risk",

    values="count",

    hole=0.5

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP USERS
# ==========================================================

st.subheader(
    "Top Goal Achievement Users"
)

top_users = (

    finance_df

    .sort_values(

        by="goal_success_probability",

        ascending=False

    )

    .head(10)

)

st.dataframe(

    top_users[

        [

            "user_id",

            "goal_success_probability",

            "goal_risk",

            "financial_health_score"

        ]

    ],

    use_container_width=True

)