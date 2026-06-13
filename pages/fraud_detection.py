from utils.load_models import *
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    paysim_df = pd.read_csv(
        "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/datasets/paysim_clean.csv"
    )

    return paysim_df


paysim_df = load_data()

# ==========================================================
# LOAD MODEL
# ==========================================================

fraud_model = joblib.load(
    "models/fraud_xgb_model.pkl"
)

# ==========================================================
# HEADER
# ==========================================================

st.title("🛡️ Fraud Detection Dashboard")

st.markdown(
    """
    AI-powered fraud detection system
    using XGBoost trained on PaySim.
    """
)

st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================

total_transactions = len(paysim_df)

fraud_transactions = int(
    paysim_df["isFraud"].sum()
)

fraud_rate = round(
    (
        fraud_transactions
        /
        total_transactions
    ) * 100,
    4
)

avg_fraud_probability = round(
    paysim_df[
        "fraud_probability"
    ].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Fraud Cases",
        f"{fraud_transactions:,}"
    )

with col3:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.4f}%"
    )

with col4:

    st.metric(
        "Avg Fraud Probability",
        f"{avg_fraud_probability:.2f}%"
    )

# ==========================================================
# TRANSACTION INPUT
# ==========================================================

st.subheader(
    "Real-Time Fraud Prediction"
)

col1, col2 = st.columns(2)

with col1:

    transaction_type = st.selectbox(

        "Transaction Type",

        [
            "CASH_IN",
            "CASH_OUT",
            "TRANSFER",
            "PAYMENT",
            "DEBIT"
        ]

    )

    amount = st.number_input(

        "Amount",

        min_value=0.0,

        value=10000.0

    )

    oldbalanceOrg = st.number_input(

        "Old Balance Origin",

        min_value=0.0,

        value=50000.0

    )

    newbalanceOrig = st.number_input(

        "New Balance Origin",

        min_value=0.0,

        value=40000.0

    )

with col2:

    oldbalanceDest = st.number_input(

        "Old Balance Destination",

        min_value=0.0,

        value=10000.0

    )

    newbalanceDest = st.number_input(

        "New Balance Destination",

        min_value=0.0,

        value=20000.0

    )

# ==========================================================
# TYPE ENCODING
# ==========================================================

type_mapping = {

    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4

}

type_encoded = type_mapping[
    transaction_type
]

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

orig_balance_diff = (
    oldbalanceOrg -
    newbalanceOrig
)

dest_balance_diff = (
    newbalanceDest -
    oldbalanceDest
)

high_amount_flag = (
    1 if amount > 200000 else 0
)

amount_log = np.log1p(
    amount
)

# ==========================================================
# PREDICTION
# ==========================================================

if st.button(
    "Detect Fraud"
):

    input_data = pd.DataFrame({

        'step': [1],

        'type_encoded':
        [type_encoded],

        'amount':
        [amount],

        'oldbalanceOrg':
        [oldbalanceOrg],

        'newbalanceOrig':
        [newbalanceOrig],

        'oldbalanceDest':
        [oldbalanceDest],

        'newbalanceDest':
        [newbalanceDest],

        'orig_balance_diff':
        [orig_balance_diff],

        'dest_balance_diff':
        [dest_balance_diff],

        'high_amount_flag':
        [high_amount_flag],

        'amount_log':
        [amount_log]

    })

    prediction = fraud_model.predict(
        input_data
    )[0]

    probability = (

        fraud_model

        .predict_proba(
            input_data
        )[0][1]

        * 100

    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{probability:.2f}%"
        )

    with col2:
        
        st.subheader(
            "Prediction")

        if prediction == 1:

            st.error(
                "Fraud Detected"
            )

        else:

            st.success(
                "Legitimate Transaction"
            )

    # ======================================================
    # GAUGE
    # ======================================================

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability,

            title={
                'text':
                "Fraud Probability"
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
                        'range':[0,40],
                        'color':'green'
                    },

                    {
                        'range':[40,70],
                        'color':'orange'
                    },

                    {
                        'range':[70,100],
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

    # ======================================================
    # RISK CATEGORY
    # ======================================================
    
    st.subheader(
        "Fraud Risk Assessment")

    if probability >= 80:

        st.error(
            "High Fraud Risk"
        )

    elif probability >= 50:

        st.warning(
            "Medium Fraud Risk"
        )

    else:

        st.success(
            "Low Fraud Risk"
        )

# ==========================================================
# FRAUD RISK DISTRIBUTION
# ==========================================================

st.subheader(
    "Fraud Risk Distribution"
)

risk_counts = (

    paysim_df

    ["fraud_risk"]

    .value_counts()

    .reset_index()

)

fig = px.bar(

    risk_counts,

    x="fraud_risk",

    y="count"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TRANSACTION TYPE ANALYSIS
# ==========================================================

st.subheader(
    "Transaction Type Analysis"
)

type_counts = (

    paysim_df

    ["type"]

    .value_counts()

    .reset_index()

)

fig = px.pie(

    type_counts,

    names="type",

    values="count",

    hole=0.5

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FRAUD VS LEGITIMATE
# ==========================================================

st.subheader(
    "Fraud Distribution"
)

fraud_counts = (

    paysim_df

    ["isFraud"]

    .value_counts()

    .reset_index()

)

fig = px.bar(

    fraud_counts,

    x="isFraud",

    y="count"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FRAUD PROBABILITY DISTRIBUTION
# ==========================================================

st.subheader(
    "Fraud Probability Distribution"
)

fig = px.histogram(

    paysim_df,

    x="fraud_probability",

    nbins=50

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP FRAUD TRANSACTIONS
# ==========================================================

st.subheader(
    "Top Suspicious Transactions"
)

top_fraud = (

    paysim_df

    .sort_values(

        by="fraud_probability",

        ascending=False

    )

    .head(20)

)

display_cols = [

    "type",

    "amount",

    "fraud_probability",

    "fraud_risk"

]

st.dataframe(

    top_fraud[
        display_cols
    ],

    use_container_width=True

)

# ==========================================================
# FRAUD INSIGHTS
# ==========================================================

st.subheader(
    "Fraud Intelligence"
)

high_risk_count = len(

    paysim_df[

        paysim_df[
            "fraud_risk"
        ] == "High Risk"

    ]

)

st.info(
    f"High Risk Transactions: {high_risk_count:,}"
)

avg_amount = round(

    paysim_df[
        "amount"
    ].mean(),

    2

)

st.info(
    f"Average Transaction Amount: ${avg_amount:,.2f}"
)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.subheader(
    "Download Fraud Report"
)

csv = top_fraud.to_csv(
    index=False
)

st.download_button(

    label="Download Report",

    data=csv,

    file_name=
    "fraud_detection_report.csv",

    mime="text/csv"

)