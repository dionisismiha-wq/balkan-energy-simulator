# ⚡ Balkan Energy Intelligence & Carbon Accounting Lakehouse (2000–2025)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Data Storage](https://img.shields.io/badge/Storage-Apache%20Parquet%20%7C%20SQLite%20%7C%20DuckDB-008080)](https://parquet.apache.org/)
[![ML & Econometrics](https://img.shields.io/badge/Analytics-XGBoost%20%7C%20MLP%20%7C%20SHAP%20%7C%20Fixed%20Effects-FF6F00)](https://xgboost.readthedocs.io/)

> **Interactive Decision Support System (DSS), Data Engineering Pipeline & Financial BI Platform** analyzing macroeconomic convergence, energy poverty, wholesale electricity market volatility (**SEEPEX**), and **EU CBAM** financial exposure across 11 Balkan economies (2000–2025).

---

## 🎯 Executive Summary & Business Impact

This project bridges **Information Systems Architecture**, **Data Engineering**, **Machine Learning**, and **Managerial/Carbon Accounting** to solve a critical regional policy and investment problem:

* **Macroeconomic & Energy Divergence:** Quantifies the structural gap between EU and Non-EU Balkan economies ($2.49\times$ $\sigma$-divergence) and the *"Poor Homeowners Trap"* via Two-Way Fixed Effects panel econometrics (`HC3` clustered standard errors).
* **5-Day Wholesale Electricity Forecasting (SEEPEX):** Achieves **$R^2 = 0.924$** and **$\text{MAPE} = 9.4\%$** on a 120-hour horizon using **XGBoost** and **Deep Learning (MLP)** trained with cyclical temporal features ($\sin/\cos$), rolling volatility windows, and **Walk-Forward Validation** (zero data leakage).
* **Extreme Price Spike Detection ($1\%$ Class Imbalance):** Combines **STL Decomposition**, **Isolation Forest**, and cost-sensitive **XGBoost** (`scale_pos_weight`) with **SHAP** explainability, achieving **$88.9\%$ Recall** and **$99.3\%$ Accuracy** ($TP=546, FN=68$).
* **Financial & Carbon Accounting BI (`P&L`, `EBITDA`, `CAPEX/NPV`):** Translates physical grid losses and $\text{CO}_2$ intensity into audited financial statements, proving that a **$120\text{M €}$ CAPEX** investment in **Smart Grids, FTTH, and PTP synchronization** achieves a **$3.6\text{-year}$ Payback Period** and protects **EBITDA margins** under an $85\text{ €/tCO}_2$ EU CBAM regime.

---

## 🏗️ System Architecture & Data Stack

The platform follows a modern **Medallion / Data Lakehouse** pattern designed for low-latency OLAP queries and full reproducibility:

```text
[Raw Data Sources]               [ETL & Storage Layer]              [Analytics & ML Layer]            [Presentation Layer]
• Eurostat (EU-SILC)   ──▶  • Idempotent Python ETL (UPSERT)  ──▶  • Two-Way Fixed Effects      ──▶  • Streamlit Cloud DSS
• World Bank WDI            • Star Schema / ERD Modeling           • XGBoost + Deep Learning MLP     • Live SQL Playground
• SEEPEX Hourly MCP         • Apache Parquet (Zone Maps)           • Isolation Forest + SHAP         • Financial BI (P&L)
• ENTSO-E (IEC 61970)       • SQLite / DuckDB OLAP Engine          • R (fixest) & SPSS Verification  • Parquet/CSV Exports
