# Intraday Liquidity Monitoring using Machine Learning

Machine learning pipeline for monitoring and explaining intraday liquidity movements using payment transaction data.

---

## Business Problem

Banks are required to monitor **intraday liquidity** to ensure they can settle payment obligations throughout the day. Failure to manage intraday liquidity can lead to settlement failures, counterparty credit risk, and regulatory penalties under frameworks such as **Basel III / BCBS 248**.

Key challenges include:
- Detecting **anomalous liquidity positions** caused by unusual payment flows
- Identifying **clusters of payments** that contribute to liquidity stress
- Providing **root cause explanations** for unexpected liquidity shortfalls

This project applies machine learning to automate anomaly detection, payment clustering, and root cause analysis on a large-scale synthetic payment transaction dataset.

---

## Dataset

A **synthetic dataset of 500,000 payment transactions** was generated to simulate realistic intraday payment flows. Each record includes:

| Feature | Description |
| --- | --- |
| `payment_system` | The payment network used (e.g. CHAPS, SWIFT, FEDWIRE) |
| `settlement_time` | Timestamp of the transaction settlement |
| `transaction_amount` | Value of the payment (USD) |
| `counterparty_type` | Type of counterparty (e.g. bank, corporate, central bank) |

Derived features computed during the pipeline include rolling liquidity positions, net settlement balances, and time-of-day payment volumes.

---

## Approach

The end-to-end pipeline processes raw payment data through four stages:

```
Payment Transactions
      │
      ▼
Liquidity Calculation   ──► Compute net positions, rolling balances, intraday exposure
      │
      ▼
Anomaly Detection       ──► Flag unusual liquidity events using Isolation Forest
      │
      ▼
Payment Clustering      ──► Group transactions by behaviour using K-Means
      │
      ▼
Root Cause Analysis     ──► Explain anomalies by cluster and payment features
```

---

## Models

| Model | Purpose |
| --- | --- |
| **Isolation Forest** | Detect liquidity anomalies — identifies transactions or time windows where liquidity positions deviate significantly from expected behaviour |
| **K-Means Clustering** | Identify payment clusters — groups transactions by payment system, counterparty type, size, and timing to surface behavioural patterns |
| **Aggregation Model** | Compute liquidity position — calculates net intraday liquidity across payment systems and counterparties at each time step |

---

## Project Structure

```
intraday-liquidity-monitoring-ml/
├── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pandas, scikit-learn, numpy

### Installation

```bash
git clone https://github.com/tkorzhan1995/intraday-liquidity-monitoring-ml.git
cd intraday-liquidity-monitoring-ml
pip install -r requirements.txt
```

---

## Results

The pipeline demonstrates that machine learning can effectively:
- Flag intraday liquidity anomalies with high precision
- Cluster payment flows into interpretable behavioural groups
- Provide actionable root cause insights for treasury and risk teams

---

## References

- [BCBS 248 – Monitoring tools for intraday liquidity management](https://www.bis.org/publ/bcbs248.htm)
- [Basel III – Liquidity Coverage Ratio](https://www.bis.org/basel_framework/chapter/LCR/10.htm)

