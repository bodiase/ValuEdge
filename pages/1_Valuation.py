# Generated from: 1_Valuation.ipynb
# Converted at: 2026-04-21T19:45:35.542Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# PAGE CONFIG
st.set_page_config(page_title="Valuation", page_icon="📈", layout="wide")


# IMPORTANT: CLASS LABEL MAPPING
CLASS_LABELS = {
    0: "Overvalued",
    1: "Fairly Valued",
    2: "Undervalued",
}

CLASS_COLORS = {
    "Overvalued": "red",
    "Fairly Valued": "orange",
    "Undervalued": "green",
}


# FILE PATH HELPERS
BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_PATHS = {
    "model": [
        BASE_DIR / "models" / "final_model.pkl",
        BASE_DIR / "final_model.pkl",
    ],
    "feature_cols": [
        BASE_DIR / "models" / "feature_cols.pkl",
        BASE_DIR / "feature_cols.pkl",
    ],
    "valuation_data": [
        BASE_DIR / "data" / "ticker_history_input.csv",
        BASE_DIR / "ticker_history_input.csv",
    ],
    "coefficients": [
        BASE_DIR / "data" / "valuation_final_model_coefficients.csv",
        BASE_DIR / "valuation_final_model_coefficients.csv",
    ],
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


# LOAD DATA / MODEL
@st.cache_resource
def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_cols(feature_cols_path: Path):
    with open(feature_cols_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_csv(csv_path: Path):
    return pd.read_csv(csv_path)


model_path = require_path("model")
feature_cols_path = require_path("feature_cols")
valuation_data_path = require_path("valuation_data")
coefficients_path = require_path("coefficients")

pipeline = load_model(model_path)
feature_cols = load_feature_cols(feature_cols_path)
valuation_df = load_csv(valuation_data_path)
coef_df = load_csv(coefficients_path)


# VALIDATION
required_cols = {"ticker", "year", *feature_cols}
missing_in_data = required_cols - set(valuation_df.columns)
if missing_in_data:
    st.error(f"ticker_history_input.csv is missing required columns: {sorted(missing_in_data)}")
    st.stop()

required_coef_cols = {"feature", "coefficient_class_0", "coefficient_class_1", "coefficient_class_2"}
missing_in_coef = required_coef_cols - set(coef_df.columns)
if missing_in_coef:
    st.error(
        f"valuation_final_model_coefficients.csv is missing required columns: {sorted(missing_in_coef)}"
    )
    st.stop()

coef_df = coef_df[coef_df["feature"].isin(feature_cols)].copy()
coef_df = coef_df.set_index("feature").reindex(feature_cols).reset_index()

if coef_df["feature"].isna().any():
    st.error("Coefficient file does not align cleanly with feature_cols.pkl.")
    st.stop()


# HELPER FUNCTIONS
def format_pct(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}%}"


def format_num(x, decimals=3):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def label_with_badge(label: str):
    color = CLASS_COLORS.get(label, "blue")
    return f":{color}[**{label}**]"


def get_latest_row_for_ticker(df: pd.DataFrame, ticker: str) -> pd.Series:
    ticker_df = df[df["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("year")
    return ticker_df.iloc[-1]


def generate_prediction(selected_row: pd.Series):
    X = pd.DataFrame([selected_row[feature_cols].astype(float).to_dict()])
    pred_class = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0]
    proba_df = pd.DataFrame({
        "class_id": pipeline.named_steps["model"].classes_,
        "probability": proba
    })
    proba_df["label"] = proba_df["class_id"].map(CLASS_LABELS)
    proba_df = proba_df.sort_values("probability", ascending=False).reset_index(drop=True)
    return pred_class, proba_df


def compute_driver_chart_data(selected_row: pd.Series, pred_class: int):
    """
    Build an approximate 'what drove this result?' view by combining:
    standardized feature value for the selected company
    x class-specific coefficient from the exported CSV

    This uses:
    - scaler parameters from the saved pipeline
    - coefficients from the exported CSV
    """
    scaler = pipeline.named_steps["scaler"]

    feature_values = selected_row[feature_cols].astype(float).values
    standardized = (feature_values - scaler.mean_) / scaler.scale_

    coef_col = f"coefficient_class_{pred_class}"
    coefs = coef_df[coef_col].values

    contribution = standardized * coefs

    driver_df = pd.DataFrame({
        "feature": feature_cols,
        "raw_value": feature_values,
        "standardized_value": standardized,
        "coefficient": coefs,
        "contribution": contribution,
    })

    driver_df["abs_contribution"] = driver_df["contribution"].abs()
    driver_df = driver_df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    return driver_df


def prettify_feature_name(feature: str) -> str:
    return (
        feature.replace("_", " ")
        .replace("roa", "ROA")
        .title()
        .replace("Roa", "ROA")
    )


def make_interpretation(label: str, top_drivers: pd.DataFrame) -> str:
    top_features = [prettify_feature_name(f) for f in top_drivers["feature"].head(3).tolist()]
    features_text = ", ".join(top_features[:-1]) + f", and {top_features[-1]}" if len(top_features) >= 3 else ", ".join(top_features)

    if label == "Undervalued":
        return (
            f"The model classifies this company as **Undervalued**. "
            f"The strongest drivers of this result were {features_text}."
        )
    elif label == "Fairly Valued":
        return (
            f"The model classifies this company as **Fairly Valued**. "
            f"The model sees a balanced profile overall, with the biggest influence coming from {features_text}."
        )
    elif label == "Overvalued":
        return (
            f"The model classifies this company as **Overvalued**. "
            f"The strongest drivers of this result were {features_text}."
        )
    else:
        return (
            f"The model produced a valuation result. "
            f"The strongest drivers were {features_text}."
        )


# TICKER SELECTION
tickers = sorted(valuation_df["ticker"].dropna().unique().tolist())

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = tickers[0] if tickers else None

st.title("📈 Valuation")
st.caption("Model-based valuation assessment using the final saved valuation model.")

col_select, col_button = st.columns([3, 1])

with col_select:
    selected_ticker = st.selectbox(
        "Select a company ticker",
        options=tickers,
        index=tickers.index(st.session_state.selected_ticker) if st.session_state.selected_ticker in tickers else 0,
        help="Choose one of the available tickers in the dataset."
    )

with col_button:
    st.write("")
    st.write("")
    analyze_clicked = st.button("Analyze", use_container_width=True)

if analyze_clicked:
    st.session_state.selected_ticker = selected_ticker

selected_ticker = st.session_state.selected_ticker

if selected_ticker is None:
    st.warning("No ticker available in the dataset.")
    st.stop()

selected_row = get_latest_row_for_ticker(valuation_df, selected_ticker)
selected_year = int(selected_row["year"])

# RUN MODEL
pred_class, proba_df = generate_prediction(selected_row)
pred_label = CLASS_LABELS.get(pred_class, f"Class {pred_class}")
driver_df = compute_driver_chart_data(selected_row, pred_class)

# SECTION 1: HEADER / CONTEXT
st.markdown(f"### Selected Company: `{selected_ticker}`")
st.write(f"Using the most recent available observation in the app dataset: **{selected_year}**")


# SECTION 2: VALUATION RESULT
st.markdown("## 1) Valuation Result")

top_prob = float(proba_df.iloc[0]["probability"])

result_col1, result_col2, result_col3 = st.columns([1.2, 1, 1])

with result_col1:
    st.metric(
        label="Valuation Classification",
        value=pred_label,
    )

with result_col2:
    st.metric(
        label="Model Confidence",
        value=f"{top_prob:.1%}",
    )

with result_col3:
    st.metric(
        label="Model Year Used",
        value=str(selected_year),
    )

st.markdown("**Class probabilities**")
proba_display = proba_df[["label", "probability"]].copy()
proba_display["probability"] = proba_display["probability"].map(lambda x: f"{x:.1%}")
st.dataframe(proba_display, use_container_width=True, hide_index=True)

# SHORT INTERPRETATION
st.markdown("## 2) Interpretation")
interpretation = make_interpretation(pred_label, driver_df)
st.info(interpretation)


# KEY DRIVERS
st.markdown("## 3) What Drove This Result?")

st.caption(
    "This chart shows the strongest model drivers for the selected company. "
    "Positive and negative bars reflect how strongly each feature pushed the prediction "
    "for the predicted class."
)

top_n = 6
driver_chart = driver_df.head(top_n).copy()
driver_chart["feature_label"] = driver_chart["feature"].map(prettify_feature_name)
driver_chart = driver_chart.sort_values("contribution", ascending=True)

st.bar_chart(
    data=driver_chart.set_index("feature_label")["contribution"],
    use_container_width=True,
)

with st.expander("See driver details"):
    details_df = driver_chart[[
        "feature_label",
        "raw_value",
        "standardized_value",
        "coefficient",
        "contribution"
    ]].copy()

    details_df.columns = [
        "Feature",
        "Raw Value",
        "Standardized Value",
        "Coefficient",
        "Contribution"
    ]

    st.dataframe(
        details_df.style.format({
            "Raw Value": "{:,.4f}",
            "Standardized Value": "{:,.4f}",
            "Coefficient": "{:,.4f}",
            "Contribution": "{:,.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# NAVIGATION HINT

st.markdown("## 4) Explore More")
st.markdown(
    """
- **Peer Comparison:** See how this company compares with its peers on key financial metrics.
- **Methodology:** See the full model logic, coefficient breakdown, and feature design.
- **Risk (CAPM):** Review the company’s market-risk profile and CAPM metrics.
"""
)
