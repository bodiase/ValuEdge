# pages/3_Risk.py

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Risk Analysis (CAPM)", layout="wide")

# GLOBAL TICKER STATE
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

# COMPANY NAMES
TICKER_NAME_MAP = {
    "LLY": "Eli Lilly and Company",
    "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co., Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "AMGN": "Amgen Inc.",
    "ABT": "Abbott Laboratories",
    "TMO": "Thermo Fisher Scientific Inc.",
    "GILD": "Gilead Sciences, Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "CVS": "CVS Health Corporation",
    "BMY": "Bristol-Myers Squibb Company",
    "MDT": "Medtronic plc",
    "CI": "Cigna Group",
    "ZTS": "Zoetis Inc.",
    "SYK": "Stryker Corporation",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "HCA": "HCA Healthcare, Inc.",
    "DHR": "Danaher Corporation",
    "HUM": "Humana Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "MRNA": "Moderna, Inc.",
    "PFE": "Pfizer Inc.",
    "BIIB": "Biogen Inc.",
    "ILMN": "Illumina, Inc.",
    "EW": "Edwards Lifesciences Corporation",
    "A": "Agilent Technologies, Inc.",
    "DXCM": "DexCom, Inc.",
    "IDXX": "IDEXX Laboratories, Inc.",
    "ALGN": "Align Technology, Inc.",
}

# SIDEBAR
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Current ticker: `{ticker}`")


# FILE HELPERS
def find_file(filename):
    possible_paths = [
        Path(filename),
        Path(".") / filename,
        Path("data") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


@st.cache_data
def load_capm_data():
    csv_path = find_file("capm_risk_metrics.csv")
    if csv_path is None:
        raise FileNotFoundError("capm_risk_metrics.csv not found")
    return pd.read_csv(csv_path)


# HELPERS
def company_display_name(ticker_value):
    company_name = TICKER_NAME_MAP.get(str(ticker_value).upper(), "")
    if company_name:
        return f"{ticker_value} ({company_name})"
    return str(ticker_value)


def safe_num(x, digits=3):
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"


def classify_beta(beta):
    if pd.isna(beta):
        return "unknown"
    if beta < 0.9:
        return "defensive"
    if beta > 1.1:
        return "aggressive"
    return "market_aligned"


def classify_alpha(alpha):
    if pd.isna(alpha):
        return "unknown"
    if alpha > 0.001:
        return "positive"
    if alpha < -0.001:
        return "negative"
    return "neutral"


def classify_r2(r_squared):
    if pd.isna(r_squared):
        return "unknown"
    if r_squared > 0.6:
        return "strong"
    if r_squared >= 0.3:
        return "moderate"
    return "weak"


def build_headline(beta_case, alpha_case):
    if beta_case == "aggressive" and alpha_case == "positive":
        return "Higher market risk with evidence of rewarded risk"
    if beta_case == "aggressive" and alpha_case == "negative":
        return "Higher market risk without strong risk-adjusted reward"
    if beta_case == "defensive" and alpha_case == "positive":
        return "Defensive profile with favorable risk-adjusted performance"
    if beta_case == "defensive" and alpha_case == "negative":
        return "Defensive profile, but recent returns have not been rewarding"
    if beta_case == "market_aligned" and alpha_case == "positive":
        return "Market-like risk with modest evidence of outperformance"
    if beta_case == "market_aligned" and alpha_case == "negative":
        return "Market-like risk with weaker risk-adjusted performance"
    return "Balanced market-risk profile with no strong excess return signal"


def build_interpretation(beta, alpha, r_squared):
    beta_case = classify_beta(beta)
    alpha_case = classify_alpha(alpha)
    r2_case = classify_r2(r_squared)

    headline = build_headline(beta_case, alpha_case)

    if beta_case == "aggressive":
        beta_text = (
            f"The stock’s beta of **{beta:.3f}** is above 1.0, which means it has shown "
            "greater sensitivity to market movements than the benchmark."
        )
    elif beta_case == "defensive":
        beta_text = (
            f"The stock’s beta of **{beta:.3f}** is below 1.0, which means it has shown "
            "lower sensitivity to market movements than the benchmark."
        )
    else:
        beta_text = (
            f"The stock’s beta of **{beta:.3f}** is close to 1.0, which suggests "
            "a market-like level of systematic risk."
        )

    if alpha_case == "positive":
        alpha_text = (
            f"Its alpha of **{alpha:.4f}** is positive, suggesting the company has delivered "
            "returns above CAPM expectations on a risk-adjusted basis."
        )
    elif alpha_case == "negative":
        alpha_text = (
            f"Its alpha of **{alpha:.4f}** is negative, suggesting the company has underperformed "
            "relative to CAPM expectations on a risk-adjusted basis."
        )
    else:
        alpha_text = (
            f"Its alpha of **{alpha:.4f}** is close to zero, which suggests little clear evidence "
            "of abnormal performance relative to CAPM expectations."
        )

    if r2_case == "strong":
        r2_text = (
            f"The R-squared of **{r_squared:.3f}** indicates that market movements explain a large share "
            "of the stock’s return variation, so the CAPM interpretation is relatively strong."
        )
    elif r2_case == "moderate":
        r2_text = (
            f"The R-squared of **{r_squared:.3f}** indicates a moderate CAPM fit. "
            "Market movements matter, but they do not explain everything."
        )
    else:
        r2_text = (
            f"The R-squared of **{r_squared:.3f}** indicates a weak CAPM fit. "
            "That means company-specific factors likely play a larger role, so CAPM-based conclusions should be used more cautiously."
        )

    detail = f"{beta_text} {alpha_text} {r2_text}"
    return headline, detail


def build_takeaways(beta, alpha, r_squared):
    beta_case = classify_beta(beta)
    alpha_case = classify_alpha(alpha)
    r2_case = classify_r2(r_squared)

    takeaways = []

    if beta_case == "aggressive":
        takeaways.append(
            "The stock carries above-market systematic risk, so investors should expect larger swings than the market benchmark."
        )
    elif beta_case == "defensive":
        takeaways.append(
            "The stock carries below-market systematic risk, which can make it behave more defensively during market moves."
        )
    else:
        takeaways.append(
            "The stock’s systematic risk is broadly in line with the market, so its volatility profile is not unusually extreme."
        )

    if alpha_case == "positive":
        takeaways.append(
            "Positive alpha is a favorable signal because it suggests returns have been stronger than CAPM would predict for this level of market risk."
        )
    elif alpha_case == "negative":
        takeaways.append(
            "Negative alpha is a caution signal because it suggests the stock has not been rewarding investors enough for the level of market risk taken."
        )
    else:
        takeaways.append(
            "Alpha is close to neutral, so the stock is not showing a strong abnormal-return signal relative to CAPM expectations."
        )

    if r2_case == "strong":
        takeaways.append(
            "The CAPM fit is relatively strong, so market risk appears to explain a meaningful share of the stock’s behavior."
        )
    elif r2_case == "moderate":
        takeaways.append(
            "The CAPM fit is moderate, so beta is useful but should still be interpreted alongside firm-specific context."
        )
    else:
        takeaways.append(
            "The CAPM fit is weak, so firm-specific events and idiosyncratic factors may be more important than market exposure alone."
        )

    return takeaways


@st.cache_data(ttl=3600)
def fetch_preliminary_2026_capm_snapshot(ticker_value, market_ticker="SPY"):
    start_date = "2026-01-01"
    end_date = date.today().isoformat()

    stock_df = yf.download(
        ticker_value,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    market_df = yf.download(
        market_ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if stock_df.empty or market_df.empty:
        raise ValueError("No yfinance data returned for the requested period.")

    stock_prices = stock_df["Close"].rename("stock_close")
    market_prices = market_df["Close"].rename("market_close")

    combined = pd.concat([stock_prices, market_prices], axis=1).dropna()

    if len(combined) < 30:
        raise ValueError("Not enough overlapping 2026 price data to estimate a preliminary CAPM snapshot.")

    combined["stock_ret"] = combined["stock_close"].pct_change()
    combined["market_ret"] = combined["market_close"].pct_change()
    returns = combined[["stock_ret", "market_ret"]].dropna()

    if len(returns) < 20:
        raise ValueError("Not enough return observations to estimate a preliminary CAPM snapshot.")

    x = returns["market_ret"].values
    y = returns["stock_ret"].values

    X = np.column_stack([np.ones(len(x)), x])
    intercept, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = intercept + beta * x

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    daily_alpha = intercept
    annualized_alpha_approx = ((1 + daily_alpha) ** 252) - 1 if pd.notna(daily_alpha) else np.nan

    return {
        "start_date": start_date,
        "end_date": end_date,
        "n_obs": int(len(returns)),
        "beta": float(beta),
        "daily_alpha": float(daily_alpha),
        "annualized_alpha_approx": float(annualized_alpha_approx),
        "r_squared": float(r_squared),
        "market_ticker": market_ticker,
    }


def build_preliminary_snapshot_text(beta, alpha, r_squared):
    beta_case = classify_beta(beta)
    alpha_case = classify_alpha(alpha)
    r2_case = classify_r2(r_squared)

    if beta_case == "aggressive":
        beta_text = "The current year-to-date estimate suggests above-market sensitivity."
    elif beta_case == "defensive":
        beta_text = "The current year-to-date estimate suggests below-market sensitivity."
    else:
        beta_text = "The current year-to-date estimate suggests market-like sensitivity."

    if alpha_case == "positive":
        alpha_text = "The preliminary alpha estimate is positive."
    elif alpha_case == "negative":
        alpha_text = "The preliminary alpha estimate is negative."
    else:
        alpha_text = "The preliminary alpha estimate is close to neutral."

    if r2_case == "strong":
        r2_text = "Market movements explain a relatively large share of the variation in returns so far this year."
    elif r2_case == "moderate":
        r2_text = "Market movements explain a moderate share of return variation so far this year."
    else:
        r2_text = "The CAPM fit is weak so far this year, so this snapshot should be treated cautiously."

    return f"{beta_text} {alpha_text} {r2_text}"


# LOAD MAIN DATA
try:
    capm_df = load_capm_data()
except Exception as e:
    st.error("capm_risk_metrics.csv is missing.")
    st.exception(e)
    st.stop()

capm_row = capm_df[capm_df["ticker"].astype(str).str.upper() == str(ticker).upper()]

if capm_row.empty:
    st.warning("No CAPM risk metrics found for this ticker.")
    st.stop()

capm_row = capm_row.iloc[0]

beta = capm_row["beta"]
alpha = capm_row["alpha"]
r_squared = capm_row["r_squared"]
beta_risk_label = str(capm_row["beta_risk_label"])

headline, interpretation_text = build_interpretation(beta, alpha, r_squared)
takeaways = build_takeaways(beta, alpha, r_squared)

# PAGE
st.title("Risk Analysis (CAPM)")
st.markdown(f"### Current ticker: `{company_display_name(ticker)}`")
st.caption(
    "This page summarizes the selected company’s CAPM-based market risk profile using beta, alpha, and R-squared."
)

st.markdown("## 1) Key Risk Metrics")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Alpha", safe_num(alpha, 4))
r2.metric("Beta", safe_num(beta, 3))
r3.metric("R-squared", safe_num(r_squared, 3))
r4.metric("Beta Risk Label", beta_risk_label)

st.markdown("## 2) What These Metrics Mean")
st.markdown(
    """
- **Beta** measures how sensitive the stock has been to overall market movements  
- **Alpha** measures whether the stock outperformed or underperformed relative to CAPM expectations  
- **R-squared** measures how much of the stock’s return variation is explained by market movements  
"""
)

st.caption(
    "CAPM is a benchmark framework for asking whether a stock’s return has been high or low relative to the amount of market risk it takes."
)

st.markdown("## 3) Beta vs Market")

beta_chart_df = pd.DataFrame({
    "Category": ["Company Beta", "Market Beta"],
    "Value": [beta, 1.0],
    "Group": ["Company", "Benchmark"],
})

beta_chart = (
    alt.Chart(beta_chart_df)
    .mark_bar(size=55)
    .encode(
        x=alt.X("Category:N", title=None),
        y=alt.Y("Value:Q", title="Beta"),
        color=alt.Color(
            "Group:N",
            scale=alt.Scale(
                domain=["Company", "Benchmark"],
                range=["#1f77b4", "#9ecae1"]
            ),
            legend=alt.Legend(title=None)
        ),
        tooltip=[
            alt.Tooltip("Category:N", title="Category"),
            alt.Tooltip("Value:Q", title="Beta", format=".3f"),
        ],
    )
    .properties(height=360)
)

beta_labels = (
    alt.Chart(beta_chart_df)
    .mark_text(dy=-8, fontSize=12, color="white")
    .encode(
        x=alt.X("Category:N"),
        y=alt.Y("Value:Q"),
        text=alt.Text("Value:Q", format=".2f"),
    )
)

st.altair_chart(beta_chart + beta_labels, use_container_width=True)

st.caption(
    "A beta of 1.0 represents market-level sensitivity. Values below 1.0 suggest a more defensive profile, while values above 1.0 suggest higher market sensitivity."
)

st.markdown("## 4) Interpretation")
st.markdown(f"### {headline}")
st.write(interpretation_text)

st.markdown("## 5) Key Takeaways")
for takeaway in takeaways:
    st.write(f"- {takeaway}")

st.markdown("## 6) Preliminary 2026 CAPM Snapshot")
st.warning(
    "This snapshot uses partial 2026 market data from a public/free source and should be treated as a preliminary update only. "
    "It is not directly comparable to the full-year CAPM results shown above."
)

st.caption(
    "Method: year-to-date daily returns for the selected stock are regressed on year-to-date daily returns for SPY as a market proxy. "
    "This is intended as a live directional update, not a finalized annual estimate."
)

try:
    snapshot = fetch_preliminary_2026_capm_snapshot(str(ticker).upper(), market_ticker="SPY")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("2026 YTD Beta", safe_num(snapshot["beta"], 3))
    s2.metric("2026 YTD Daily Alpha", safe_num(snapshot["daily_alpha"], 4))
    s3.metric("2026 YTD R-squared", safe_num(snapshot["r_squared"], 3))
    s4.metric("Trading Days Used", snapshot["n_obs"])

    st.caption(
        f"Data window: **{snapshot['start_date']}** to **{snapshot['end_date']}** | "
        f"Market benchmark: **{snapshot['market_ticker']}**"
    )

    snapshot_text = build_preliminary_snapshot_text(
        snapshot["beta"],
        snapshot["daily_alpha"],
        snapshot["r_squared"],
    )
    st.info(snapshot_text)

    with st.expander("Why this section is labeled preliminary"):
        st.markdown(
            """
- It uses only **partial 2026** data rather than a completed annual window  
- It uses **daily returns from yfinance** and a simple market benchmark proxy (**SPY**)  
- The resulting beta, alpha, and R-squared estimates can move materially as more 2026 data becomes available  
- Use this section as a **current directional snapshot**, not as a finalized full-year CAPM estimate  
"""
        )

except Exception as e:
    st.info(
        "A preliminary 2026 CAPM snapshot could not be generated right now. "
        "This may happen if yfinance data is temporarily unavailable or if there are not enough 2026 observations yet."
    )
    with st.expander("Technical details"):
        st.write(str(e))

st.markdown("## 7) Explore More")
st.markdown(
    """
- **Valuation Assessment:** Review the model’s latest-year valuation signal and historical valuation context  
- **Peer Comparison:** See how the company compares with peers across profitability, leverage, liquidity, growth, and valuation metrics  
- **Methodology:** Review the project framing, data sources, and model context  
"""
)
