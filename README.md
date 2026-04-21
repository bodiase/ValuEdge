# ValuEdge  
### Integrated Equity Analysis for Valuation, Peer Comparison, and Risk

---

## Project Overview

ValuEdge is a financial analytics application that provides an integrated framework for equity analysis by combining **machine learning-based valuation**, **peer benchmarking**, and **market risk assessment**.

The application helps users evaluate whether a company appears **overvalued, fairly valued, or undervalued**, while also providing the financial context and risk insights needed to interpret that result. By bringing together multiple analytical perspectives into a single workflow, ValuEdge offers a more complete and interpretable view of a company’s financial profile.

---

## Key Features

- **Machine Learning-Based Valuation**  
  Classifies companies as Overvalued, Fairly Valued, or Undervalued using a multiclass logistic regression model.

- **Peer Comparison Framework**  
  Benchmarks companies against peer medians and relative positions across key financial metrics.

- **CAPM-Based Risk Analysis**  
  Evaluates market risk using beta, alpha, and R-squared.

- **Interpretable Model Insights**  
  Provides feature importance to explain model predictions.

- **End-to-End Analytics Pipeline**  
  Covers the full workflow from raw data to deployed interactive application.

---

## App Structure

The application consists of the following pages:

- **Home**  
  Overview of the app, its purpose, and navigation guidance.

- **Valuation**  
  Displays the model’s classification, predicted probabilities, and key drivers of the result.

- **Peer Comparison**  
  Compares the selected company to peers across profitability, valuation, leverage, liquidity, and growth metrics.

- **Risk (CAPM)**  
  Shows beta, alpha, R-squared, and a structured interpretation of market risk.

- **Methodology**  
  Explains the data pipeline, feature engineering, model selection, evaluation, and limitations.

---

## Data Sources

- **WRDS / Wharton (Compustat)**  
  Firm-level financial and accounting data used for feature construction.

- **yfinance**  
  Market data used to support valuation and contextual analysis.

- **Kenneth R. French Data Library**  
  Factor data used for CAPM-based risk modeling.

---

## Tools / Tech Stack

- Python  
- Streamlit  
- pandas  
- numpy  
- scikit-learn  

---

## Repository Structure

```text
project-repo/
│
├── streamlit_app.py        # Home / landing page
├── pages/                 # App pages
│   ├── 1_Valuation.py
│   ├── 2_Peer_Comparison.py
│   ├── 3_Risk.py
│   ├── 4_Methodology.py
│
├── data/                  # CSV data files
│   ├── model_data.csv
│   ├── peer_summary.csv
│   ├── capm_risk_metrics.csv
│   ├── ff_factors_clean.csv
│   ├── valuation_test_predictions.csv
│
├── models/                # Saved model artifacts
│   ├── final_model.pkl
│   ├── feature_cols.pkl
│
├── requirements.txt
└── README.md
```

---

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the Streamlit app:

   ```
   $ streamlit run streamlit_app.py
   ```

---

## Project Workflow Summary

**1. Data Collection**: Gathered financial and market data from WRDS, yfinance, and the Fama-French data library.

**2. Data Processing & Feature Engineering**: Cleaned data and constructed financial ratios, growth metrics, and relative features.

**3. Modeling**: Trained and evaluated multiple models; selected logistic regression for interpretability and performance.

**4. Evaluation**: Assessed model performance using classification metrics and validation techniques.

**5. Deployment**: Built an interactive Streamlit application to present results and insights.

---

## Acknowledgements
Boston University MSBA — BA870 Financial Analytics
Special thanks to Professor Peter Wysocki, our course instructor, for guidance and instruction throughout the project.

---

## Notes / Limitations
This application is an academic project designed for analytical and educational purposes only.
It should not be interpreted as financial or investment advice.

---
