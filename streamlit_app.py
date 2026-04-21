# Generated from: streamlit_app.ipynb
# Converted at: 2026-04-21T19:22:08.167Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import streamlit as st


# PAGE CONFIG
st.set_page_config(page_title="ValuEdge", page_icon="📊", layout="wide")


# HEADER
st.title("ValuEdge")
st.subheader("Integrated Equity Analysis for Valuation, Peer Context, and Risk")
st.write(
    "ValuEdge is a financial analytics application that combines **machine learning-based valuation**, "
    "**peer benchmarking**, and **CAPM-based risk analysis** into one unified workflow. "
    "It helps users evaluate whether a stock appears overvalued, fairly valued, or undervalued "
    "while also understanding the financial and market context behind that result."
)

st.divider()


# HERO SECTION
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("## Why ValuEdge?")
    st.write(
        "Most equity analysis tools focus on only one part of the decision process. "
        "ValuEdge brings together three complementary perspectives:"
    )
    st.markdown(
        """
- **Valuation** → What does the model predict?
- **Peer Comparison** → How does the company compare with similar firms?
- **Risk (CAPM)** → What kind of market risk and performance profile does it have?
"""
    )
    st.write(
        "Together, these components provide a more complete and interpretable view of a company’s financial profile."
    )

with right_col:
    st.markdown("## Quick Facts")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Core Pages", "4")
        st.metric("Data Sources", "3")
    with metric_col2:
        st.metric("Final Model", "Logistic Regression")
        st.metric("Classes", "3")


st.divider()


# WHAT THE APP DOES
st.markdown("## What the App Does")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("### 📈 Valuation")
    st.write(
        "Applies a multiclass valuation model to classify a company as "
        "**Overvalued**, **Fairly Valued**, or **Undervalued**."
    )

with info_col2:
    st.markdown("### 📊 Peer Comparison")
    st.write(
        "Benchmarks the selected company against peer medians and percentile-style "
        "relative positions across key financial metrics."
    )

with info_col3:
    st.markdown("### 📉 Risk (CAPM)")
    st.write(
        "Summarizes market sensitivity and risk-adjusted performance using "
        "**beta**, **alpha**, and **R-squared**."
    )


st.divider()


# HOW TO USE THE APP
st.markdown("## How to Use ValuEdge")

use_col1, use_col2 = st.columns([1, 1])

with use_col1:
    st.markdown(
        """
**Step 1** — Open the **Valuation** page and select a company ticker
**Step 2** — Review the model’s valuation classification and decision drivers
**Step 3** — Open **Peer Comparison** to benchmark the company against peers
**Step 4** — Open **Risk (CAPM)** to review market risk and performance
**Step 5** — Open **Methodology** for model design, evaluation, and data sources
"""
    )

with use_col2:
    st.info(
        "The selected ticker is designed to carry across pages using session state, "
        "so the app works as one connected workflow rather than as separate isolated tools."
    )


st.divider()


# PAGE GUIDE
st.markdown("## Page Guide")

guide_col1, guide_col2 = st.columns(2)

with guide_col1:
    st.markdown("### 📈 Valuation")
    st.write(
        "See the model-based valuation classification, class probabilities, "
        "and the key drivers behind the prediction."
    )

    st.markdown("### 📊 Peer Comparison")
    st.write(
        "Compare the selected company with peer medians across profitability, "
        "valuation, leverage, liquidity, and growth metrics."
    )

with guide_col2:
    st.markdown("### 📉 Risk (CAPM)")
    st.write(
        "Review beta, alpha, R-squared, and a rules-based interpretation of the "
        "company’s market-risk profile."
    )

    st.markdown("### 🧠 Methodology")
    st.write(
        "Understand the project’s modeling approach, evaluation results, feature design, "
        "data sources, and workflow from raw data to app."
    )


st.divider()


# KEY FEATURES / HIGHLIGHTS
st.markdown("## Key Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown(
        """
### 🔍 Interpretable Valuation Model
The final model is a multiclass logistic regression, allowing the app to show
not only the valuation result but also the key factors driving that result.

### 🏢 Peer-Relative Context
Companies are evaluated relative to peer benchmarks rather than in isolation,
which makes the analysis more realistic and financially meaningful.
"""
    )

with feature_col2:
    st.markdown(
        """
### 📉 Integrated Risk Lens
ValuEdge complements valuation with CAPM-based risk analysis so users can assess
market sensitivity and risk-adjusted performance alongside financial context.

### 🔗 End-to-End Analytics Workflow
The app reflects a full pipeline from raw financial and market data to feature engineering,
model training, evaluation, and interactive deployment.
"""
    )


st.divider()


# DATA SOURCES
st.markdown("## Data Sources")

source_col1, source_col2, source_col3 = st.columns(3)

with source_col1:
    st.markdown("### WRDS / Compustat")
    st.write("Firm-level accounting and financial statement data used to build the valuation feature set.")

with source_col2:
    st.markdown("### yfinance")
    st.write("Market-based inputs used to enrich the dataset with price and related market context.")

with source_col3:
    st.markdown("### Kenneth R. French Data Library")
    st.write("Factor data used to support CAPM-based market risk analysis.")


st.divider()


# WORKFLOW PREVIEW
st.markdown("## Workflow Overview")

workflow_cols = st.columns(5)
workflow_steps = [
    ("1. Data", "Collect financial and market inputs"),
    ("2. Features", "Engineer valuation and peer-relative variables"),
    ("3. Model", "Train and evaluate classification models"),
    ("4. Outputs", "Generate valuation, peer, and risk summaries"),
    ("5. App", "Deliver insights through Streamlit pages"),
]

for col, (title, desc) in zip(workflow_cols, workflow_steps):
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)


st.divider()


# CTA
st.markdown("## Start Exploring")
st.success(
    "Use the sidebar to begin with the **Valuation** page, then move to **Peer Comparison**, "
    "**Risk (CAPM)**, and **Methodology** for a complete view."
)

st.caption("ValuEdge — bringing valuation, context, and risk together in one place.")