import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="About / Methodology", layout="wide")

st.title("About / Methodology")
st.caption(
    "This page explains the project objective, data sources, modeling approach, evaluation results, and workflow behind ValueLens."
)

# FILE HELPERS
def find_first_file(candidates):
    for filename in candidates:
        possible_paths = [
            Path(filename),
            Path(".") / filename,
            Path("data") / filename,
            Path(__file__).parent.parent / filename,
            Path(__file__).parent.parent / "data" / filename,
            Path(__file__).parent.parent / "models" / filename,
        ]
        for path in possible_paths:
            if path.exists():
                return path
    return None


@st.cache_data
def load_optional_csv(candidates):
    path = find_first_file(candidates)
    if path is None:
        return None, None
    return pd.read_csv(path), path.name


# OPTIONAL FILES
comparison_df, comparison_file = load_optional_csv([
    "valuation_model_comparison.csv",
    "model_comparison_results.csv",
    "model_comparison.csv",
    "valuation_evaluation_metrics.csv",
])

pred_df, pred_file = load_optional_csv([
    "valuation_test_predictions.csv",
])

coef_df, coef_file = load_optional_csv([
    "valuation_final_model_coefficients.csv",
    "valuation_final_model_coefficients (1).csv",
])


# HELPERS
def find_column(df, candidates):
    if df is None:
        return None
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_label_series(series):
    mapping = {
        0: "Overvalued",
        1: "Fairly valued",
        2: "Undervalued",
        "0": "Overvalued",
        "1": "Fairly valued",
        "2": "Undervalued",
        "overvalued": "Overvalued",
        "fairly valued": "Fairly valued",
        "fairly_valued": "Fairly valued",
        "undervalued": "Undervalued",
    }
    return series.map(lambda x: mapping.get(x, mapping.get(str(x).lower(), x)))


def make_confusion_matrix(df):
    actual_col = find_column(df, ["actual_label", "actual", "y_true", "true_label", "label_true"])
    predicted_col = find_column(df, ["predicted_label", "predicted", "y_pred", "prediction", "label_pred"])

    if actual_col is None or predicted_col is None:
        return None

    actual = normalize_label_series(df[actual_col])
    predicted = normalize_label_series(df[predicted_col])

    order = ["Overvalued", "Fairly valued", "Undervalued"]
    cm = pd.crosstab(
        pd.Categorical(actual, categories=order),
        pd.Categorical(predicted, categories=order),
        dropna=False,
    )
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    return cm


def compute_basic_metrics(df):
    actual_col = find_column(df, ["actual_label", "actual", "y_true", "true_label", "label_true"])
    predicted_col = find_column(df, ["predicted_label", "predicted", "y_pred", "prediction", "label_pred"])

    if actual_col is None or predicted_col is None:
        return None

    actual = normalize_label_series(df[actual_col]).astype(str)
    predicted = normalize_label_series(df[predicted_col]).astype(str)

    valid_mask = actual.notna() & predicted.notna()
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) == 0:
        return None

    accuracy = (actual == predicted).mean()

    classes = sorted(set(actual.unique()).union(set(predicted.unique())))
    per_class = []
    for cls in classes:
        tp = ((actual == cls) & (predicted == cls)).sum()
        fp = ((actual != cls) & (predicted == cls)).sum()
        fn = ((actual == cls) & (predicted != cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class.append({
            "Class": cls,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    macro_f1 = pd.DataFrame(per_class)["F1"].mean()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_obs": len(actual),
        "per_class": pd.DataFrame(per_class),
    }


def extract_model_column(df):
    return find_column(df, ["model", "model_name", "estimator"])


def extract_metric_column(df, candidates):
    return find_column(df, candidates)


def format_pct(x):
    try:
        return f"{100 * float(x):.1f}%"
    except Exception:
        return "N/A"


def format_num(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


# TOP SUMMARY
top1, top2, top3, top4 = st.columns(4)
top1.metric("Final Model", "Logistic Regression")
top2.metric("Task Type", "3-Class Classification")
top3.metric("Sector Focus", "Healthcare")
top4.metric("Primary App Goal", "Valuation + Peers + Risk")

st.divider()

# SECTION 1
st.markdown("## 1) Project Objective")
st.write(
    "ValueLens is a financial analytics app designed to evaluate healthcare companies from three complementary angles: "
    "**valuation**, **peer comparison**, and **market risk**. "
    "The core modeling task is a three-class classification problem that estimates whether a company appears "
    "**Overvalued**, **Fairly valued**, or **Undervalued** based on financial and valuation features."
)

# SECTION 2
st.markdown("## 2) What the Valuation Model Predicts")
st.markdown(
    """
- **Overvalued:** the company’s feature profile looks expensive relative to the model’s learned patterns  
- **Fairly valued:** the company’s profile appears broadly aligned with the model’s learned patterns  
- **Undervalued:** the company’s profile looks relatively attractive compared with the model’s learned patterns  
"""
)
st.caption(
    "This is a classification model, not a direct intrinsic-value calculator. It is designed to identify valuation status based on patterns in financial data."
)

# SECTION 3
st.markdown("## 3) Data Sources")

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("### WRDS / Compustat")
    st.write(
        "Used offline for historical firm fundamentals and accounting data. "
        "These data supported dataset construction, feature engineering, and model training."
    )

with d2:
    st.markdown("### yfinance")
    st.write(
        "Used for public/free market data components. "
        "This supports deployment-safe market-based analytics and optional live/public updates."
    )

with d3:
    st.markdown("### Kenneth R. French Data Library")
    st.write(
        "Used for factor-based risk modeling inputs that support CAPM-style market risk analysis."
    )

st.caption(
    "The deployed app does not rely on WRDS at runtime. WRDS was used offline to build and train the project artifacts; the app runs from saved CSV/PKL outputs and public/free sources where applicable."
)

# SECTION 4
st.markdown("## 4) Feature Groups / Inputs")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("### Profitability & Efficiency")
    st.markdown(
        """
- ROA  
- Operating Margin  
- Liquidity / working-capital style indicators  
"""
    )

with f2:
    st.markdown("### Growth & Change Features")
    st.markdown(
        """
- Revenue Growth  
- Year-over-year changes in key ratios  
- Directional changes in operating performance  
"""
    )

with f3:
    st.markdown("### Relative Valuation & Peer Context")
    st.markdown(
        """
- Price to Sales  
- Price to Book  
- Peer-relative versions of selected metrics  
- Relative leverage / profitability comparisons  
"""
    )

st.caption(
    "These feature groups were designed to capture not only a company’s own financial profile, but also how that profile compares with peers."
)

# SECTION 5
st.markdown("## 5) Final Model Choice")
st.write(
    "The final model used in the app is **Logistic Regression**. "
    "It was selected because it offered a strong balance of performance, interpretability, and clean deployment. "
    "Compared with more complex alternatives, logistic regression made it easier to explain why a company was classified into a given valuation category."
)

# SECTION 6
st.markdown("## 6) Model Evaluation")

basic_metrics = compute_basic_metrics(pred_df) if pred_df is not None else None

if basic_metrics is not None:
    e1, e2, e3 = st.columns(3)
    e1.metric("Test Accuracy", format_pct(basic_metrics["accuracy"]))
    e2.metric("Macro F1", format_num(basic_metrics["macro_f1"], 3))
    e3.metric("Test Observations", str(basic_metrics["n_obs"]))
else:
    st.info(
        "Primary evaluation metrics were not computed on-page because a usable prediction file was not found or did not contain the required columns."
    )

if basic_metrics is not None:
    st.markdown("### Per-Class Performance")
    per_class_df = basic_metrics["per_class"].copy()
    st.dataframe(
        per_class_df.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# OPTIONAL: CONFUSION MATRIX
st.markdown("### OPTIONAL: Confusion Matrix")
cm = make_confusion_matrix(pred_df) if pred_df is not None else None

if cm is not None:
    cm_long = (
        cm.reset_index()
        .melt(id_vars="Actual", var_name="Predicted", value_name="Count")
    )

    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", title="Predicted Label"),
            y=alt.Y("Actual:N", title="Actual Label"),
            color=alt.Color("Count:Q", title="Count"),
            tooltip=["Actual", "Predicted", "Count"],
        )
        .properties(height=320)
    )

    text = (
        alt.Chart(cm_long)
        .mark_text(fontSize=13)
        .encode(
            x="Predicted:N",
            y="Actual:N",
            text="Count:Q",
            color=alt.condition(
                alt.datum.Count > cm_long["Count"].mean(),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    st.altair_chart(heatmap + text, use_container_width=True)
else:
    st.caption(
        "Confusion matrix not shown because `valuation_test_predictions.csv` was not found or did not contain usable actual/predicted label columns."
    )

# SECTION 7
st.markdown("## 7) Alternative Models Tested")

if comparison_df is not None:
    model_col = extract_model_column(comparison_df)
    accuracy_col = extract_metric_column(comparison_df, ["accuracy", "test_accuracy", "acc"])
    f1_col = extract_metric_column(comparison_df, ["macro_f1", "f1", "test_f1"])

    st.caption(f"Loaded comparison file: `{comparison_file}`")

    show_cols = [col for col in [model_col, accuracy_col, f1_col] if col is not None]
    comparison_display = comparison_df.copy()

    if model_col is not None and accuracy_col is not None:
        comparison_display = comparison_display.sort_values(accuracy_col, ascending=False)

    if show_cols:
        st.dataframe(
            comparison_display[show_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)

    st.write(
        "Alternative models were tested to compare predictive performance and deployment tradeoffs. "
        "Even when another model looked competitive, logistic regression remained attractive because it was easier to explain and integrate into the app."
    )
else:
    st.write(
        "Multiple models were evaluated during development, but the final deployment version emphasizes logistic regression because it provided a good tradeoff between predictive usefulness and interpretability."
    )

# SECTION 8
st.markdown("## 8) Model Explanation / Coefficient-Based Insight")

if coef_df is not None:
    st.caption(f"Loaded coefficient file: `{coef_file}`")

    available_class_cols = [
        col for col in coef_df.columns
        if col.startswith("coefficient_class_")
    ]

    if "feature" in coef_df.columns and available_class_cols:
        selected_class = st.selectbox(
            "Choose a valuation class to inspect",
            options=available_class_cols,
            format_func=lambda x: {
                "coefficient_class_0": "Overvalued",
                "coefficient_class_1": "Fairly valued",
                "coefficient_class_2": "Undervalued",
            }.get(x, x),
        )

        coef_display = coef_df[["feature", selected_class]].copy()
        coef_display["abs_coef"] = coef_display[selected_class].abs()
        coef_display = coef_display.sort_values("abs_coef", ascending=False).head(12)

        coef_chart = (
            alt.Chart(coef_display)
            .mark_bar()
            .encode(
                x=alt.X(f"{selected_class}:Q", title="Coefficient"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", alt.Tooltip(selected_class, format=".4f")],
            )
            .properties(height=360)
        )

        st.altair_chart(coef_chart, use_container_width=True)
        st.caption(
            "These coefficients help show which features most strongly influence the model for the selected class. "
            "Larger absolute values indicate stronger directional influence in the classification rule."
        )
    else:
        st.info("Coefficient file was found, but it does not have the expected structure.")
else:
    st.write(
        "The app can also explain valuation outputs using exported coefficient data from the final logistic regression model. "
        "If the coefficient CSV is present in the app folder, this section can visualize class-specific model drivers."
    )

# SECTION 9
st.markdown("## 9) Workflow Summary")

with st.expander("Show pipeline summary"):
    st.markdown(
        """
1. **Collect historical financial and market data** using course-approved sources  
2. **Construct cleaned modeling datasets** and engineer ratios, change features, and peer-relative features  
3. **Train and compare multiple models** offline using the prepared training data  
4. **Export deployment-safe artifacts** such as CSV files and saved model objects  
5. **Deploy the app** using saved files plus public/free sources where appropriate  
"""
    )

# SECTION 10
st.markdown("## 10) Limitations")

with st.expander("Show limitations and caveats"):
    st.markdown(
        """
- The valuation model is a **classification tool**, not a direct intrinsic value calculator  
- Model outputs depend on the selected features, training period, and class definitions  
- Peer-based comparisons are only as strong as the chosen comparison group  
- CAPM-style risk metrics simplify reality and do not capture all sources of uncertainty  
- Public/live add-ons should be treated as **supplemental snapshots**, not replacements for the main offline analysis  
"""
    )

st.divider()
st.caption(
    "Recommended supporting files for this page: `valuation_test_predictions.csv`, "
    "`valuation_model_comparison.csv` (or similar comparison file), and "
    "`valuation_final_model_coefficients.csv`."
)
