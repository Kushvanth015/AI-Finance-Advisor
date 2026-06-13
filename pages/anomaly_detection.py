import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
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
# LOAD MODELS
# ==========================================================

iso_model = joblib.load(
    "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/models/isolation_forest_model.pkl"
)

anomaly_scaler = joblib.load(
    "https://huggingface.co/Kushvanth05/AI-Finance-Advisor-Models/resolve/main/scalers/anomaly_scaler.pkl"
)

# ==========================================================
# HEADER
# ==========================================================

st.title("🚨 Anomaly Detection Dashboard")

st.markdown(
    """
    Unsupervised anomaly detection system using
    Isolation Forest to identify suspicious
    and unusual transaction behavior.
    """
)

st.divider()

# ==========================================================
# KPI SECTION
# ==========================================================

total_transactions = len(paysim_df)

total_anomalies = int(
    paysim_df["is_anomaly"].sum()
)

anomaly_rate = round(
    (
        total_anomalies /
        total_transactions
    ) * 100,
    2
)

avg_score = round(
    paysim_df["anomaly_score"].mean(),
    4
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Anomalies",
        f"{total_anomalies:,}"
    )

with col3:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

with col4:

    st.metric(
        "Average Score",
        avg_score
    )

# ==========================================================
# REAL-TIME ANOMALY CHECK
# ==========================================================

st.subheader(
    "Real-Time Anomaly Detection"
)

col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "Transaction Amount(Sender side)",
        min_value=0.0,
        value=10000.0
    )

    oldbalanceOrg = st.number_input(
        "Old Origin Balance(Sender side)",
        min_value=0.0,
        value=50000.0
    )

    newbalanceOrig = st.number_input(
        "New Origin Balance(Sender side)",
        min_value=0.0,
        value=40000.0
    )

with col2:

    oldbalanceDest = st.number_input(
        "Old Destination Balance(Receiver side)",
        min_value=0.0,
        value=10000.0
    )

    newbalanceDest = st.number_input(
        "New Destination Balance(Receiver side)",
        min_value=0.0,
        value=20000.0
    )

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

amount_log = np.log1p(
    amount
)

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("Detect Anomaly"):

    # ======================================================
    # BUSINESS RULE VALIDATION
    # ======================================================

    rule_anomaly = False

    reasons = []

    expected_new_origin = (
        oldbalanceOrg - amount
    )

    expected_new_destination = (
        oldbalanceDest + amount
    )

    origin_diff = abs(
        expected_new_origin -
        newbalanceOrig
    )

    destination_diff = abs(
        expected_new_destination -
        newbalanceDest
    )

    # Sender gained money after transfer

    if newbalanceOrig > oldbalanceOrg:

        rule_anomaly = True

        reasons.append(
            "Sender balance increased after transfer"
        )

    # Receiver lost money after receiving

    if newbalanceDest < oldbalanceDest:

        rule_anomaly = True

        reasons.append(
            "Receiver balance decreased after transfer"
        )

    # Sender mismatch

    if origin_diff > 1:

        rule_anomaly = True

        reasons.append(
            "Sender balance mismatch"
        )

    # Receiver mismatch

    if destination_diff > 1:

        rule_anomaly = True

        reasons.append(
            "Receiver balance mismatch"
        )

    # =====================================
    # CONTINUE WITH MODEL
    # =====================================

    input_df = pd.DataFrame({

        'amount':[amount],

        'oldbalanceOrg':[oldbalanceOrg],

        'newbalanceOrig':[newbalanceOrig],

        'oldbalanceDest':[oldbalanceDest],

        'newbalanceDest':[newbalanceDest],

        'orig_balance_diff':[orig_balance_diff],

        'dest_balance_diff':[dest_balance_diff],

        'amount_log':[amount_log]

    })

    scaled_data = anomaly_scaler.transform(
        input_df
    )

    prediction = iso_model.predict(
        scaled_data
    )[0]

    anomaly_score = (
        iso_model.score_samples(
            scaled_data
        )[0]
    )
    
    scaled_data = (
        anomaly_scaler.transform(
            input_df
        )
    )

    prediction = (
        iso_model.predict(
            scaled_data
        )[0]
    )

    anomaly_score = (
        iso_model.score_samples(
            scaled_data
        )[0]
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Anomaly Score",
            round(
                anomaly_score,
                4
            )
        )

    with col2:

        # ======================================================
        # FINAL DECISION ENGINE
        # ======================================================
        
        st.subheader("Prediction")

        if rule_anomaly:

            st.error(
                "🚨 Anomaly Detected"
            )

            st.error(
                "High Risk Transaction"
            )

            for reason in reasons:

                st.warning(reason)

        elif prediction == -1:

            st.error(
                "🚨 Anomaly Detected"
            )

        else:

            st.success(
                "✅ Normal Transaction"
            )
    
    # ======================================================
    # ANOMALY GAUGE
    # ======================================================

    gauge_value = max(
        0,
        min(
            100,
            (
                abs(anomaly_score)
                * 100
            )
        )
    )

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=gauge_value,

            title={
                'text':
                "Anomaly Risk"
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
    
    st.subheader("Risk Level")

    if rule_anomaly:
        
        st.error(
            "🚨 High Risk"
        )

    elif prediction == -1:

        st.error(
            "🚨 High Risk"
        )

    elif prediction == -1:

        st.error(
            "🚨 High Risk"
        )

    elif anomaly_score < -0.40:

        st.warning(
            "⚠️ Medium Risk"
        )

    else:

        st.success(
            "✅ Low Risk"
        )

# ==========================================================
# ANOMALY RISK DISTRIBUTION
# ==========================================================

st.subheader(
    "Anomaly Risk Distribution"
)

risk_counts = (

    paysim_df

    ["anomaly_risk"]

    .value_counts()

    .reset_index()
)

fig = px.bar(

    risk_counts,

    x="anomaly_risk",

    y="count"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# ANOMALY SCORE DISTRIBUTION
# ==========================================================

st.subheader(
    "Anomaly Score Distribution"
)

fig = px.histogram(

    paysim_df,

    x="anomaly_score",

    nbins=50

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# FRAUD VS ANOMALY
# ==========================================================

st.subheader(
    "Fraud vs Anomaly"
)

comparison = pd.crosstab(

    paysim_df["isFraud"],

    paysim_df["is_anomaly"]

)

st.dataframe(
    comparison,
    use_container_width=True
)

# ==========================================================
# TOP ANOMALIES
# ==========================================================

st.subheader(
    "Top Suspicious Transactions"
)

top_anomalies = (

    paysim_df

    .sort_values(
        by="anomaly_score"
    )

    .head(20)

)

display_cols = [

    "type",

    "amount",

    "oldbalanceOrg",

    "newbalanceOrig",

    "anomaly_score",

    "anomaly_risk"

]

st.dataframe(

    top_anomalies[
        display_cols
    ],

    use_container_width=True

)

# ==========================================================
# ANOMALY ANALYTICS
# ==========================================================

st.subheader(
    "Anomaly Insights"
)

avg_amount = round(

    paysim_df[
        paysim_df[
            "is_anomaly"
        ] == 1

    ]["amount"]

    .mean(),

    2

)

st.info(
    f"Average Anomaly Amount: ${avg_amount:,.2f}"
)

high_risk_count = len(

    paysim_df[

        paysim_df[
            "anomaly_risk"
        ] == "High Risk"

    ]

)

st.info(
    f"High Risk Anomalies: {high_risk_count:,}"
)

# ==========================================================
# AMOUNT VS SCORE
# ==========================================================

st.subheader(
    "Amount vs Anomaly Score"
)

sample_df = paysim_df.sample(
    min(10000, len(paysim_df)),
    random_state=42
)

fig = px.scatter(

    sample_df,

    x="amount",

    y="anomaly_score",

    color="anomaly_risk"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

st.subheader(
    "Download Anomaly Report"
)

csv = top_anomalies.to_csv(
    index=False
)

st.download_button(

    label="Download Report",

    data=csv,

    file_name=
    "anomaly_detection_report.csv",

    mime="text/csv"

)