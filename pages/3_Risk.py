# Generated from: 3_Risk.ipynb
# Converted at: 2026-04-21T19:45:50.554Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# PAGE CONFIG
st.set_page_config(page_title="Risk (CAPM)", page_icon="📉", layout="wide")


# FILE PATHS
BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_PATHS = {
    "capm_risk_metrics": [
        BASE_DIR / "data" / "capm_risk_metrics.csv",
        BASE_DIR / "capm_risk_metrics.csv",
    ]
}


def first_existing_path(path_list):
    for path in path_list:
        if path.exists():
            return path
    return None


def require_path(key: str) -> Path:
    path = first_existing_path(CANDIDATE_PATHS[key])
    if path is None:
        searched = "\n".join(str(p) for p in CANDIDATE_PATHS[key])
        st.error(
            f"Could not find the required file for '{key}'.\n\n"
            f"Searched these locations:\n{searched}"
        )
        st.stop()
    return path


# LOAD DATA
@st.cache_data
def load_csv(csv_path: Path):
    return pd.read_csv(csv_path)


capm_path = require_path("capm_risk_metrics")
risk_df = load_csv(capm_path)


# REQUIRED COLUMNS
required_columns = {
    "ticker",
    "alpha",
    "beta",
    "r_squared",
    "beta_risk_label",
}

missing_columns = required_columns - set(risk_df.columns)
if missing_columns:
    st.error(f"capm_risk_metrics.csv is missing required columns: {sorted(missing_columns)}")
    st.stop()


# HELPERS
def format_decimal(x, decimals=3):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def format_percent(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}%}"


def get_selected_row(df: pd.DataFrame, ticker: str) -> pd.Series:
    row_df = df[df["ticker"] == ticker].copy()
    if row_df.empty:
        st.error(f"No CAPM metrics found for ticker: {ticker}")
        st.stop()
    return row_df.iloc[0]


def classify_beta(beta):
    if pd.isna(beta):
        return "unknown"
    if beta > 1.10:
        return "high"
    if beta < 0.90:
        return "low"
    return "market_like"


def classify_alpha(alpha):
    if pd.isna(alpha):
        return "unknown"
    if alpha > 0:
        return "positive"
    if alpha < 0:
        return "negative"
    return "neutral"


def classify_r_squared(r_squared):
    if pd.isna(r_squared):
        return "unknown"
    if r_squared > 0.60:
        return "strong"
    if r_squared >= 0.30:
        return "moderate"
    return "weak"


def build_takeaways(row: pd.Series):
    beta = row["beta"]
    alpha = row["alpha"]
    r_squared = row["r_squared"]
    risk_label = row["beta_risk_label"]

    beta_case = classify_beta(beta)
    alpha_case = classify_alpha(alpha)
    r2_case = classify_r_squared(r_squared)

    takeaways = []

    # Market risk
    if beta_case == "high":
        takeaways.append(
            f"The stock appears more volatile than the market, with beta above 1.0 and a risk label of **{risk_label}**."
        )
    elif beta_case == "low":
        takeaways.append(
            f"The stock appears less volatile than the market, with beta below 1.0 and a risk label of **{risk_label}**."
        )
    elif beta_case == "market_like":
        takeaways.append(
            f"The stock’s market sensitivity is broadly in line with the market overall, with beta close to 1.0."
        )

    # Risk-adjusted performance
    if alpha_case == "positive":
        takeaways.append(
            "The stock has generated positive abnormal return relative to CAPM expectations."
        )
    elif alpha_case == "negative":
        takeaways.append(
            "The stock has underperformed relative to CAPM expectations on a risk-adjusted basis."
        )
    elif alpha_case == "neutral":
        takeaways.append(
            "The stock’s risk-adjusted performance is broadly neutral relative to CAPM expectations."
        )

    # Model fit
    if r2_case == "strong":
        takeaways.append(
            "CAPM explains a large share of the stock’s return variation, so the beta estimate is relatively informative."
        )
    elif r2_case == "moderate":
        takeaways.append(
            "CAPM explains a moderate share of return variation, so beta is useful but should not be interpreted too narrowly."
        )
    elif r2_case == "weak":
        takeaways.append(
            "CAPM explains only a limited share of return variation, so market beta may not fully capture this stock’s behavior."
        )

    # Combined summary
    if beta_case == "high" and alpha_case == "positive":
        takeaways.append(
            "Overall, this looks like a higher-risk stock that has still delivered positive risk-adjusted performance."
        )
    elif beta_case == "high" and alpha_case == "negative":
        takeaways.append(
            "Overall, this looks like a higher-risk stock that has not been rewarded with positive risk-adjusted performance."
        )
    elif beta_case == "low" and alpha_case == "positive":
        takeaways.append(
            "Overall, this looks like a lower-risk stock that has still produced positive risk-adjusted performance."
        )
    elif beta_case == "low" and alpha_case == "negative":
        takeaways.append(
            "Overall, this looks like a lower-risk stock, but recent risk-adjusted performance has been weak."
        )
    elif beta_case == "market_like" and alpha_case == "positive":
        takeaways.append(
            "Overall, this stock’s market risk is fairly typical, but returns have outperformed CAPM expectations."
        )
    elif beta_case == "market_like" and alpha_case == "negative":
        takeaways.append(
            "Overall, this stock’s market risk is fairly typical, but returns have underperformed CAPM expectations."
        )

    if not takeaways:
        takeaways.append("CAPM metrics are available, but the overall risk profile is not strongly differentiated.")

    return takeaways[:4]


def build_beta_chart_df(beta_value):
    return pd.DataFrame(
        {
            "Series": ["Selected Stock", "Market Reference"],
            "Beta": [beta_value, 1.0],
        }
    )


# TICKER STATE
tickers = sorted(risk_df["ticker"].dropna().unique().tolist())

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = tickers[0] if tickers else None

current_ticker = st.session_state.selected_ticker


# HEADER
st.title("📉 Risk (CAPM)")
st.caption("Review CAPM-based market risk metrics for the selected company.")

col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown(f"### Current selected ticker: `{current_ticker}`")

with col_b:
    override_ticker = st.selectbox(
        "Optional ticker override",
        options=tickers,
        index=tickers.index(current_ticker) if current_ticker in tickers else 0,
        help="Use the shared selected ticker by default, or override it here if needed."
    )

use_override = st.checkbox("Use override ticker on this page", value=False)

if use_override:
    selected_ticker = override_ticker
else:
    selected_ticker = current_ticker

if selected_ticker is None:
    st.warning("No ticker available in the dataset.")
    st.stop()

selected_row = get_selected_row(risk_df, selected_ticker)

beta = selected_row["beta"]
alpha = selected_row["alpha"]
r_squared = selected_row["r_squared"]
risk_label = selected_row["beta_risk_label"]
takeaways = build_takeaways(selected_row)
beta_chart_df = build_beta_chart_df(beta)


# SECTION 1: SELECTED TICKER + INTRO
st.markdown("## 1) Selected Company")
st.write(
    f"This page summarizes **CAPM-based risk metrics** for **{selected_ticker}**. "
    f"It focuses on market sensitivity, risk-adjusted performance, and how informative CAPM appears to be for this stock."
)


# SECTION 2: KEY RISK METRICS
st.markdown("## 2) Key Risk Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Beta", format_decimal(beta, 3))

with metric_col2:
    st.metric("Alpha", format_decimal(alpha, 3))

with metric_col3:
    st.metric("R-squared", format_percent(r_squared, 1))

with metric_col4:
    st.metric("Risk Label", str(risk_label))


# SECTION 3: BETA VISUALIZATION
st.markdown("## 3) Beta vs Market")
st.caption(
    "A beta of 1.0 represents market-level sensitivity. "
    "Values above 1.0 indicate greater sensitivity to market movements, while values below 1.0 indicate lower sensitivity."
)

beta_chart = beta_chart_df.set_index("Series")
st.bar_chart(beta_chart, use_container_width=True)

reference_text = ""
if classify_beta(beta) == "high":
    reference_text = "This stock’s beta is above the market reference level of 1.0."
elif classify_beta(beta) == "low":
    reference_text = "This stock’s beta is below the market reference level of 1.0."
else:
    reference_text = "This stock’s beta is close to the market reference level of 1.0."

st.write(reference_text)


# SECTION 4: INTERPRETATION
st.markdown("## 4) Key Takeaways")
st.info(
    "These takeaways are generated using simple rules based on beta, alpha, R-squared, and the provided risk label."
)

for takeaway in takeaways:
    st.markdown(f"- {takeaway}")


# SECTION 5: EXPLORE MORE
st.markdown("## 5) Explore More")
st.markdown(
    """
- **Valuation:** See the model-based valuation classification for this company.
- **Peer Comparison:** Compare this company’s financial profile with peer medians.
- **Methodology:** Review how risk metrics and model outputs were constructed.
"""
)