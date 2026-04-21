# Generated from: 4_Methodology.ipynb
# Converted at: 2026-04-21T19:46:05.807Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix


# PAGE CONFIG
st.set_page_config(page_title="Methodology", page_icon="🧠", layout="wide")


# FILE PATHS
BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_PATHS = {
    "model_comparison": [
        BASE_DIR / "data" / "valuation_model_comparison_results.csv",
        BASE_DIR / "valuation_model_comparison_results.csv",
    ],
    "test_predictions": [
        BASE_DIR / "data" / "valuation_test_predictions.csv",
        BASE_DIR / "data" / "valuation_test_predictions (1).csv",
        BASE_DIR / "valuation_test_predictions.csv",
        BASE_DIR / "valuation_test_predictions (1).csv",
    ],
    "coefficients": [
        BASE_DIR / "data" / "valuation_final_model_coefficients.csv",
        BASE_DIR / "data" / "valuation_final_model_coefficients (1).csv",
        BASE_DIR / "valuation_final_model_coefficients.csv",
        BASE_DIR / "valuation_final_model_coefficients (1).csv",
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


# LOAD DATA
@st.cache_data
def load_csv(csv_path: Path):
    return pd.read_csv(csv_path)


comparison_path = require_path("model_comparison")
predictions_path = require_path("test_predictions")
coefficients_path = require_path("coefficients")

comparison_df = load_csv(comparison_path)
predictions_df = load_csv(predictions_path)
coefficients_df = load_csv(coefficients_path)


# REQUIRED COLUMNS
comparison_required = {
    "model",
    "train_accuracy",
    "test_accuracy",
    "train_f1",
    "test_f1",
}

predictions_required = {
    "ticker",
    "year",
    "valuation_score",
    "actual_valuation_label",
    "predicted_valuation_label",
}

coefficients_required = {
    "feature",
    "coefficient_class_0",
    "coefficient_class_1",
    "coefficient_class_2",
}

missing_comparison = comparison_required - set(comparison_df.columns)
missing_predictions = predictions_required - set(predictions_df.columns)
missing_coefficients = coefficients_required - set(coefficients_df.columns)

if missing_comparison:
    st.error(f"valuation_model_comparison_results.csv is missing columns: {sorted(missing_comparison)}")
    st.stop()

if missing_predictions:
    st.error(f"valuation_test_predictions.csv is missing columns: {sorted(missing_predictions)}")
    st.stop()

if missing_coefficients:
    st.error(f"valuation_final_model_coefficients.csv is missing columns: {sorted(missing_coefficients)}")
    st.stop()


# HELPERS
CLASS_LABELS = {
    0: "Overvalued",
    1: "Fairly Valued",
    2: "Undervalued",
}


def format_decimal(x, decimals=3):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def format_percent(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}%}"


def prettify_feature_name(feature: str) -> str:
    return (
        feature.replace("_", " ")
        .replace("roa", "ROA")
        .title()
        .replace("Roa", "ROA")
    )


def label_name(label):
    return CLASS_LABELS.get(label, str(label))


def build_confusion_matrix_df(pred_df: pd.DataFrame):
    labels = [0, 1, 2]
    cm = confusion_matrix(
        pred_df["actual_valuation_label"],
        pred_df["predicted_valuation_label"],
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label_name(x)}" for x in labels],
        columns=[f"Predicted: {label_name(x)}" for x in labels],
    )
    return cm_df


def build_class_summary(pred_df: pd.DataFrame):
    labels = [0, 1, 2]
    rows = []

    for label in labels:
        actual_mask = pred_df["actual_valuation_label"] == label
        total_actual = actual_mask.sum()

        if total_actual == 0:
            recall = None
        else:
            correct = (
                (pred_df["actual_valuation_label"] == label)
                & (pred_df["predicted_valuation_label"] == label)
            ).sum()
            recall = correct / total_actual

        pred_mask = pred_df["predicted_valuation_label"] == label
        total_pred = pred_mask.sum()

        if total_pred == 0:
            precision = None
        else:
            correct_pred = (
                (pred_df["actual_valuation_label"] == label)
                & (pred_df["predicted_valuation_label"] == label)
            ).sum()
            precision = correct_pred / total_pred

        rows.append(
            {
                "Class": label_name(label),
                "Support": total_actual,
                "Precision": precision,
                "Recall": recall,
            }
        )

    return pd.DataFrame(rows)


def build_feature_groups():
    return pd.DataFrame(
        [
            {
                "Feature Group": "Profitability",
                "Example Features": "ROA, Operating Margin",
                "Why It Matters": "Captures operating strength and return efficiency."
            },
            {
                "Feature Group": "Leverage / Liquidity",
                "Example Features": "Debt to Assets, Current Ratio",
                "Why It Matters": "Helps assess financial health and balance-sheet flexibility."
            },
            {
                "Feature Group": "Valuation Multiples",
                "Example Features": "Price to Sales, Price to Book",
                "Why It Matters": "Measures how richly or cheaply the market prices the firm."
            },
            {
                "Feature Group": "Growth",
                "Example Features": "Revenue Growth",
                "Why It Matters": "Adds business momentum and expansion context."
            },
            {
                "Feature Group": "Change Features",
                "Example Features": "ROA Change, Operating Margin Change, Debt to Assets Change",
                "Why It Matters": "Captures recent improvement or deterioration in fundamentals."
            },
            {
                "Feature Group": "Peer-Relative Features",
                "Example Features": "ROA Rel, Operating Margin Rel, Debt to Assets Rel, Price to Sales Rel, Price to Book Rel",
                "Why It Matters": "Measures the company relative to peer benchmarks, not just on a standalone basis."
            },
        ]
    )


def build_data_sources_df():
    return pd.DataFrame(
        [
            {
                "Source": "WRDS / Wharton (Compustat)",
                "What It Provided": "Firm-level financial statement data and accounting fundamentals.",
                "Why It Matters": "Used to build the structured financial feature set for valuation."
            },
            {
                "Source": "yfinance",
                "What It Provided": "Market-based inputs such as price history and related market fields.",
                "Why It Matters": "Used to enrich the company dataset with market context."
            },
            {
                "Source": "Kenneth R. French Data Library",
                "What It Provided": "Factor data used for the CAPM/risk side of the app.",
                "Why It Matters": "Supports market-risk analysis and factor-based interpretation."
            },
        ]
    )


def build_pipeline_df():
    return pd.DataFrame(
        [
            {"Step": "1. Raw Data", "Description": "Collected accounting and market inputs from source datasets."},
            {"Step": "2. Data Cleaning", "Description": "Standardized fields, aligned time periods, and prepared usable company-year observations."},
            {"Step": "3. Feature Engineering", "Description": "Built core, change-based, and peer-relative valuation features."},
            {"Step": "4. Label Construction", "Description": "Converted the valuation score into three classes: Overvalued, Fairly Valued, and Undervalued."},
            {"Step": "5. Model Training", "Description": "Tested multiple classification models using a time-based train/test split."},
            {"Step": "6. Model Selection", "Description": "Selected the final model based on out-of-sample macro F1 and overall interpretability."},
            {"Step": "7. App Deployment", "Description": "Saved the final artifacts and connected them to Streamlit pages for interactive use."},
        ]
    )


def build_alternative_models_text():
    if comparison_df.empty:
        return "No model comparison results available."

    sorted_models = comparison_df.sort_values(["test_f1", "test_accuracy"], ascending=False).reset_index(drop=True)
    best_model = sorted_models.iloc[0]["model"]

    lines = []
    for _, row in sorted_models.iterrows():
        lines.append(
            f"- **{row['model']}**: Test Accuracy = {row['test_accuracy']:.3f}, Test Macro F1 = {row['test_f1']:.3f}"
        )

    summary = (
        f"The final selected model was **{best_model}**, chosen because it delivered the strongest out-of-sample "
        f"macro F1 while also remaining highly interpretable for a finance-focused app."
    )
    return summary, lines


def build_feature_importance_table(coef_df: pd.DataFrame):
    coef_df = coef_df.copy()
    coef_df["max_abs_coef"] = coef_df[
        ["coefficient_class_0", "coefficient_class_1", "coefficient_class_2"]
    ].abs().max(axis=1)
    coef_df["Feature"] = coef_df["feature"].map(prettify_feature_name)

    out = coef_df[["Feature", "max_abs_coef"]].sort_values("max_abs_coef", ascending=False).reset_index(drop=True)
    out.columns = ["Feature", "Max Absolute Coefficient"]
    return out.head(10)


# DERIVED OBJECTS
best_model_row = comparison_df.sort_values(["test_f1", "test_accuracy"], ascending=False).reset_index(drop=True).iloc[0]
best_model_name = best_model_row["model"]

comparison_display = comparison_df.copy()
comparison_display["train_accuracy"] = comparison_display["train_accuracy"].map(lambda x: f"{x:.3f}")
comparison_display["test_accuracy"] = comparison_display["test_accuracy"].map(lambda x: f"{x:.3f}")
comparison_display["train_f1"] = comparison_display["train_f1"].map(lambda x: f"{x:.3f}")
comparison_display["test_f1"] = comparison_display["test_f1"].map(lambda x: f"{x:.3f}")

cm_df = build_confusion_matrix_df(predictions_df)
class_summary_df = build_class_summary(predictions_df)
feature_groups_df = build_feature_groups()
data_sources_df = build_data_sources_df()
pipeline_df = build_pipeline_df()
feature_importance_df = build_feature_importance_table(coefficients_df)
alt_summary, alt_lines = build_alternative_models_text()


# HEADER
st.title("🧠 Methodology")
st.caption("How the project moves from raw financial data to valuation outputs and app deployment.")


# SECTION 1: PROJECT OVERVIEW
st.markdown("## 1) Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.write(
        "This project builds a financial analytics app that estimates whether a stock appears "
        "**Overvalued**, **Fairly Valued**, or **Undervalued** based on structured financial, "
        "market, and peer-relative features."
    )
    st.write(
        "The goal is not to produce a price target. Instead, the model generates a **relative valuation classification** "
        "based on a firm’s financial profile and how that profile compares with peers."
    )

with col2:
    st.metric("Final Selected Model", best_model_name)
    st.metric("Classes", "3")
    st.metric("Test Observations", int(len(predictions_df)))


# SECTION 2: MODELING APPROACH
st.markdown("## 2) Modeling Approach")

col1, col2 = st.columns([2, 1])

with col1:
    st.write(
        "The final valuation model is a **multiclass logistic regression** model trained on company-year observations. "
        "The classes are encoded as:"
    )
    st.markdown(
        """
- **0 = Overvalued**
- **1 = Fairly Valued**
- **2 = Undervalued**
"""
    )
    st.write(
        "A time-based train/test split was used so that the model was evaluated on later periods rather than randomly mixed observations. "
        "This makes the setup more realistic for applied forecasting and deployment."
    )

with col2:
    st.info(
        "Why logistic regression?\n\n"
        "- Strong out-of-sample performance\n"
        "- Interpretable coefficients\n"
        "- Well suited for structured financial data"
    )


# SECTION 3: FEATURES / INPUTS
st.markdown("## 3) Features / Inputs")
st.write(
    "The model uses a mix of core financial metrics, change-based features, and peer-relative features. "
    "Each group contributes different information about valuation, quality, financial health, and business momentum."
)

st.dataframe(feature_groups_df, use_container_width=True, hide_index=True)

with st.expander("Top coefficient-based driver features"):
    st.dataframe(
        feature_importance_df.style.format({"Max Absolute Coefficient": "{:,.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


# SECTION 4: MODEL OUTPUT & INTERPRETATION
st.markdown("## 4) Model Output & Interpretation")

col1, col2 = st.columns([1, 1])

with col1:
    st.write(
        "The output of the valuation model is a **class prediction**, not a direct valuation multiple or price target."
    )
    st.markdown(
        """
- **Overvalued**: the company’s feature profile looks expensive relative to the model’s learned patterns
- **Fairly Valued**: the company appears broadly in line with expected valuation conditions
- **Undervalued**: the company’s feature profile looks relatively attractive given the model inputs
"""
    )

with col2:
    st.info(
        "On the Valuation page, the app shows:\n\n"
        "- the predicted valuation class\n"
        "- model confidence / class probabilities\n"
        "- key drivers of the decision"
    )


# SECTION 5: MODEL EVALUATION
st.markdown("## 5) Model Evaluation")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Final Test Accuracy", format_percent(best_model_row["test_accuracy"], 1))

with metric_col2:
    st.metric("Final Test Macro F1", format_decimal(best_model_row["test_f1"], 3))

with metric_col3:
    st.metric("Final Train Accuracy", format_percent(best_model_row["train_accuracy"], 1))

with metric_col4:
    st.metric("Final Train Macro F1", format_decimal(best_model_row["train_f1"], 3))

st.write(
    "The final model was selected based primarily on **out-of-sample macro F1**, which is appropriate for a multiclass setup "
    "because it gives balanced weight across the three valuation classes."
)

st.markdown("### Class-level summary")
class_summary_display = class_summary_df.copy()
class_summary_display["Precision"] = class_summary_display["Precision"].map(lambda x: format_percent(x, 1))
class_summary_display["Recall"] = class_summary_display["Recall"].map(lambda x: format_percent(x, 1))
st.dataframe(class_summary_display, use_container_width=True, hide_index=True)

# OPTIONAL: Confusion matrix
with st.expander("Optional: Confusion matrix"):
    st.write(
        "The confusion matrix below is computed directly from `valuation_test_predictions.csv`. "
        "Rows are the true classes and columns are the predicted classes."
    )
    st.dataframe(cm_df, use_container_width=True)


# SECTION 6: ALTERNATIVE MODELS TESTED
st.markdown("## 6) Alternative Models Tested")

st.write(
    "Several classification models were evaluated before selecting the final model. "
    "This section shows that the model choice was intentional rather than arbitrary."
)

st.dataframe(comparison_display, use_container_width=True, hide_index=True)

st.write(alt_summary)
for line in alt_lines:
    st.markdown(line)


# SECTION 7: DATA SOURCES
st.markdown("## 7) Data Sources")

st.dataframe(data_sources_df, use_container_width=True, hide_index=True)


# SECTION 8: PIPELINE / WORKFLOW
st.markdown("## 8) Pipeline / Workflow")

st.dataframe(pipeline_df, use_container_width=True, hide_index=True)


# SECTION 9: LIMITATIONS
st.markdown("## 9) Limitations")

with st.expander("Show limitations"):
    st.markdown(
        """
- The model is trained on historical company-year observations, so future market conditions may differ from the training period.
- The output is a **classification signal**, not a direct estimate of intrinsic value or a guaranteed investment recommendation.
- Relative valuation depends on the quality of the engineered peer and market features used in the pipeline.
- CAPM and peer benchmarking are simplifications; real-world valuation is influenced by many qualitative and forward-looking factors not fully captured here.
- Even strong out-of-sample metrics do not eliminate the possibility of regime shifts, changing fundamentals, or model degradation over time.
"""
    )


# CTA
st.markdown("## Explore More")
st.markdown(
    """
- **Valuation:** See the model-based valuation classification for a selected company.
- **Peer Comparison:** Compare company metrics against peer medians and percentile positions.
- **Risk (CAPM):** Review beta, alpha, and CAPM-based market-risk interpretation.
"""
)