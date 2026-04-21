# Generated from: 2_Peer_Comparison.ipynb
# Converted at: 2026-04-21T19:45:43.047Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# PAGE CONFIG
st.set_page_config(page_title="Peer Comparison", page_icon="📊", layout="wide")


# FILE PATHS
BASE_DIR = Path(__file__).resolve().parents[1]

CANDIDATE_PATHS = {
    "peer_summary": [
        BASE_DIR / "data" / "peer_summary.csv",
        BASE_DIR / "peer_summary.csv",
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


peer_summary_path = require_path("peer_summary")
peer_df = load_csv(peer_summary_path)


# REQUIRED COLUMNS
required_columns = {
    "ticker",
    "year",
    "roa", "roa_median", "roa_pct",
    "operating_margin", "operating_margin_median", "operating_margin_pct",
    "debt_to_assets", "debt_to_assets_median", "debt_to_assets_pct",
    "current_ratio", "current_ratio_median", "current_ratio_pct",
    "price_to_sales", "price_to_sales_median", "price_to_sales_pct",
    "price_to_book", "price_to_book_median", "price_to_book_pct",
    "revenue_growth", "revenue_growth_median", "revenue_growth_pct",
}

missing_columns = required_columns - set(peer_df.columns)
if missing_columns:
    st.error(f"peer_summary.csv is missing required columns: {sorted(missing_columns)}")
    st.stop()


# HELPERS
def format_decimal(x, decimals=3):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def format_percentile(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def format_difference(company_val, peer_val, decimals=3):
    if pd.isna(company_val) or pd.isna(peer_val):
        return "N/A"
    diff = company_val - peer_val
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:,.{decimals}f}"


def pct_bucket_text(x):
    if pd.isna(x):
        return "not available"
    if x >= 0.80:
        return "top quintile"
    if x >= 0.60:
        return "above average"
    if x >= 0.40:
        return "middle of the peer group"
    if x >= 0.20:
        return "below average"
    return "bottom quintile"


def compare_standard(company_val, peer_val, high_threshold=0.10):
    if pd.isna(company_val) or pd.isna(peer_val):
        return "in_line"
    if peer_val == 0:
        if company_val > 0:
            return "above"
        if company_val < 0:
            return "below"
        return "in_line"

    ratio_diff = abs(company_val - peer_val) / max(abs(peer_val), 1e-9)
    if ratio_diff <= high_threshold:
        return "in_line"
    return "above" if company_val > peer_val else "below"


def compare_inverse(company_val, peer_val, high_threshold=0.10):
    # For metrics where LOWER is usually better, like debt_to_assets or valuation multiples.
    if pd.isna(company_val) or pd.isna(peer_val):
        return "in_line"
    if peer_val == 0:
        if company_val < 0:
            return "better"
        if company_val > 0:
            return "worse"
        return "in_line"

    ratio_diff = abs(company_val - peer_val) / max(abs(peer_val), 1e-9)
    if ratio_diff <= high_threshold:
        return "in_line"
    return "better" if company_val < peer_val else "worse"


def get_latest_row_for_ticker(df: pd.DataFrame, ticker: str) -> pd.Series:
    ticker_df = df[df["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("year")
    return ticker_df.iloc[-1]


def build_summary_table(selected_row: pd.Series) -> pd.DataFrame:
    metrics = [
        ("ROA", "roa", "roa_median", "roa_pct"),
        ("Operating Margin", "operating_margin", "operating_margin_median", "operating_margin_pct"),
        ("Debt to Assets", "debt_to_assets", "debt_to_assets_median", "debt_to_assets_pct"),
        ("Current Ratio", "current_ratio", "current_ratio_median", "current_ratio_pct"),
        ("Price to Sales", "price_to_sales", "price_to_sales_median", "price_to_sales_pct"),
        ("Price to Book", "price_to_book", "price_to_book_median", "price_to_book_pct"),
        ("Revenue Growth", "revenue_growth", "revenue_growth_median", "revenue_growth_pct"),
    ]

    rows = []
    for label, company_col, peer_col, pct_col in metrics:
        company_val = selected_row[company_col]
        peer_val = selected_row[peer_col]
        pct_val = selected_row[pct_col]

        rows.append({
            "Metric": label,
            "Company Value": company_val,
            "Peer Median": peer_val,
            "Difference": company_val - peer_val if pd.notna(company_val) and pd.notna(peer_val) else np.nan,
            "Peer Percentile": pct_val,
        })

    return pd.DataFrame(rows)


def build_chart_df(selected_row: pd.Series) -> pd.DataFrame:
    metrics = [
        ("ROA", "roa", "roa_median"),
        ("Operating Margin", "operating_margin", "operating_margin_median"),
        ("Debt to Assets", "debt_to_assets", "debt_to_assets_median"),
        ("Current Ratio", "current_ratio", "current_ratio_median"),
        ("Price to Sales", "price_to_sales", "price_to_sales_median"),
        ("Price to Book", "price_to_book", "price_to_book_median"),
    ]

    rows = []
    for label, company_col, peer_col in metrics:
        rows.append({
            "Metric": label,
            "Company": selected_row[company_col],
            "Peer Median": selected_row[peer_col],
        })

    return pd.DataFrame(rows)


def build_takeaways(selected_row: pd.Series):
    takeaways = []

    # Profitability
    profitability_signals = []
    roa_cmp = compare_standard(selected_row["roa"], selected_row["roa_median"])
    opm_cmp = compare_standard(selected_row["operating_margin"], selected_row["operating_margin_median"])

    if roa_cmp == "above":
        profitability_signals.append("ROA is above the peer median")
    elif roa_cmp == "below":
        profitability_signals.append("ROA is below the peer median")

    if opm_cmp == "above":
        profitability_signals.append("operating margin is above the peer median")
    elif opm_cmp == "below":
        profitability_signals.append("operating margin is below the peer median")

    if roa_cmp == "above" and opm_cmp == "above":
        takeaways.append(
            f"Profitability looks stronger than peers. The company ranks in the {pct_bucket_text(selected_row['roa_pct'])} on ROA and in the {pct_bucket_text(selected_row['operating_margin_pct'])} on operating margin."
        )
    elif roa_cmp == "below" and opm_cmp == "below":
        takeaways.append(
            f"Profitability looks weaker than peers. Both ROA and operating margin sit below peer medians."
        )
    elif profitability_signals:
        takeaways.append(
            "Profitability is mixed relative to peers: " + "; ".join(profitability_signals) + "."
        )

    # Balance sheet / liquidity
    leverage_cmp = compare_inverse(selected_row["debt_to_assets"], selected_row["debt_to_assets_median"])
    liquidity_cmp = compare_standard(selected_row["current_ratio"], selected_row["current_ratio_median"])

    if leverage_cmp == "better" and liquidity_cmp == "above":
        takeaways.append(
            f"The balance-sheet profile looks relatively strong, with lower leverage than peers and stronger liquidity."
        )
    elif leverage_cmp == "worse" and liquidity_cmp == "below":
        takeaways.append(
            f"The balance-sheet profile looks weaker than peers, with higher leverage and weaker liquidity."
        )
    elif leverage_cmp == "better":
        takeaways.append(
            f"Leverage looks more conservative than peers, with debt-to-assets below the peer median."
        )
    elif leverage_cmp == "worse":
        takeaways.append(
            f"Leverage looks heavier than peers, with debt-to-assets above the peer median."
        )
    elif liquidity_cmp == "above":
        takeaways.append(
            f"Liquidity looks stronger than peers, with the current ratio above the peer median."
        )
    elif liquidity_cmp == "below":
        takeaways.append(
            f"Liquidity looks weaker than peers, with the current ratio below the peer median."
        )

    # Valuation
    ps_cmp = compare_inverse(selected_row["price_to_sales"], selected_row["price_to_sales_median"])
    pb_cmp = compare_inverse(selected_row["price_to_book"], selected_row["price_to_book_median"])

    if ps_cmp == "better" and pb_cmp == "better":
        takeaways.append(
            f"Valuation multiples are below peer medians, suggesting the stock trades at a relative discount."
        )
    elif ps_cmp == "worse" and pb_cmp == "worse":
        takeaways.append(
            f"Valuation multiples are above peer medians, suggesting the stock trades at a richer valuation than peers."
        )
    elif ps_cmp == "better" or pb_cmp == "better":
        takeaways.append(
            f"Valuation looks somewhat cheaper than peers on at least one major multiple."
        )
    elif ps_cmp == "worse" or pb_cmp == "worse":
        takeaways.append(
            f"Valuation looks somewhat richer than peers on at least one major multiple."
        )

    # Growth
    growth_cmp = compare_standard(selected_row["revenue_growth"], selected_row["revenue_growth_median"])
    if growth_cmp == "above":
        takeaways.append(
            f"Revenue growth is stronger than peers, which may help justify a premium valuation if sustained."
        )
    elif growth_cmp == "below":
        takeaways.append(
            f"Revenue growth is weaker than peers, which may make strong valuation multiples harder to justify."
        )

    # Keep concise
    if not takeaways:
        takeaways.append("Most major metrics appear broadly in line with peer medians.")
    return takeaways[:4]


# TICKER STATE
tickers = sorted(peer_df["ticker"].dropna().unique().tolist())

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = tickers[0] if tickers else None

current_ticker = st.session_state.selected_ticker

# HEADER
st.title("📊 Peer Comparison")
st.caption("Compare the selected company to peer medians and peer-relative positions.")

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

selected_row = get_latest_row_for_ticker(peer_df, selected_ticker)
selected_year = int(selected_row["year"])

summary_table = build_summary_table(selected_row)
chart_df = build_chart_df(selected_row)
takeaways = build_takeaways(selected_row)

# SECTION 1: SELECTED TICKER + INTRO
st.markdown("## 1) Selected Company")
st.write(
    f"This page benchmarks **{selected_ticker}** against peer medians using the most recent available observation in the dataset: **{selected_year}**."
)

# SECTION 2: SUMMARY TABLE
st.markdown("## 2) Peer Comparison Summary")

display_df = summary_table.copy()
display_df["Company Value"] = display_df["Company Value"].map(lambda x: format_decimal(x, 3))
display_df["Peer Median"] = display_df["Peer Median"].map(lambda x: format_decimal(x, 3))
display_df["Difference"] = summary_table.apply(
    lambda row: format_difference(row["Company Value"], row["Peer Median"], 3), axis=1
)
display_df["Peer Percentile"] = display_df["Peer Percentile"].map(format_percentile)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# SECTION 3: CHARTS
st.markdown("## 3) Company vs Peer Visuals")
st.caption("These charts compare the selected company with the peer median across key profitability, balance-sheet, and valuation metrics.")

metric_options = chart_df["Metric"].tolist()
selected_metrics = st.multiselect(
    "Choose metrics to visualize",
    options=metric_options,
    default=metric_options,
)

filtered_chart_df = chart_df[chart_df["Metric"].isin(selected_metrics)].copy()

if not filtered_chart_df.empty:
    company_series = filtered_chart_df.set_index("Metric")["Company"]
    peer_series = filtered_chart_df.set_index("Metric")["Peer Median"]

    chart_wide = pd.concat(
        [company_series.rename("Company"), peer_series.rename("Peer Median")],
        axis=1
    )

    st.bar_chart(chart_wide, use_container_width=True)
else:
    st.info("Select at least one metric to display the comparison chart.")

# SECTION 4: KEY TAKEAWAYS
st.markdown("## 4) Key Takeaways")
st.info(
    "These takeaways are generated using simple rules based on how the selected company compares with peer medians and peer percentiles."
)

for takeaway in takeaways:
    st.markdown(f"- {takeaway}")

# OPTIONAL CTA
st.markdown("## 5) Explore More")
st.markdown(
    """
- **Valuation:** See the model-based valuation classification for this company.
- **Risk (CAPM):** Review beta, alpha, and other market-risk metrics.
- **Methodology:** See how peer benchmarks and model inputs were built.
"""
)