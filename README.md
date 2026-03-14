# Intraday Liquidity Monitoring – ML Pipeline

Machine learning pipeline for monitoring and explaining intraday liquidity
movements in banking payment systems using transaction-level data.

---

## Business Context

Banks are required to monitor **intraday liquidity** to ensure they have
sufficient funds to settle all payment obligations throughout the day.
Liquidity changes continuously as payments flow through systems such as
**TARGET2**, **SWIFT**, internal settlement platforms, and securities
settlement systems.

Failure to maintain adequate intraday liquidity can result in payment
gridlock, reputational damage, and regulatory breaches. This project
provides an automated, data-driven pipeline that:

- Tracks the running intraday liquidity position in near-real time.
- Detects abnormal liquidity movements using machine learning.
- Clusters payments by behavioural pattern to explain *why* a liquidity
  event occurred.

---

## Project Structure

```
intraday-liquidity-monitoring-ml/
├── data/
│   ├── raw/                    # Raw transaction CSV
│   │   └── transactions.csv
│   ├── processed/              # Pipeline outputs
│   └── generate_data.py        # Synthetic dataset generator
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Load, clean, aggregate, liquidity position
│   ├── feature_engineering.py  # ML feature matrices
│   ├── anomaly_detection.py    # Isolation Forest anomaly detector
│   ├── clustering.py           # K-Means payment clusterer
│   └── visualization.py        # matplotlib figures
├── notebooks/
│   └── exploratory_analysis.ipynb
├── figures/                    # Saved plots
├── requirements.txt
└── README.md
```

---

## Dataset

The pipeline consumes a transaction-level CSV with the following schema:

| Column | Type | Description |
|---|---|---|
| `payment_id` | string | Unique payment identifier |
| `timestamp` | datetime | Time the payment entered the system |
| `settlement_time` | datetime | Actual settlement time |
| `payment_system` | string | TARGET2 / SWIFT / INTERNAL / SECURITIES |
| `counterparty_type` | string | BANK / CORPORATE / CENTRAL_BANK / CCP |
| `direction` | string | INFLOW or OUTFLOW |
| `amount` | float | Payment amount (local currency) |

A synthetic dataset (2,000 transactions, one settlement day) is generated
by `data/generate_data.py` and models realistic intraday activity patterns
including a simulated afternoon stress period.

---

## Methodology

### 1 – Data Processing (`src/data_processing.py`)

- Validates schema and cleans raw transaction records.
- Aggregates payments into **5-minute time buckets** computing total
  inflow, outflow, net flow, transaction count, and average size.
- Calculates the **running intraday liquidity position** as:

  ```
  position(t) = opening_balance + Σ net_flow(i) for i ≤ t
  ```

### 2 – Feature Engineering (`src/feature_engineering.py`)

- **Bucket-level features**: rolling statistics, lag features,
  liquidity-change percentage, outflow/inflow ratio, time-of-day encoding.
- **Transaction-level features**: log-amount, time-of-day, direction flag,
  categorical encodings for payment system and counterparty type.

### 3 – Anomaly Detection (`src/anomaly_detection.py`)

Uses **Isolation Forest** (scikit-learn) to identify 5-minute buckets
with unusual liquidity behaviour. The algorithm isolates observations by
randomly partitioning the feature space; anomalies require fewer splits and
receive lower anomaly scores.

Key hyperparameters:
- `contamination = 0.05` (expected 5% anomaly rate)
- `n_estimators = 100`

### 4 – Payment Clustering (`src/clustering.py`)

Uses **K-Means** (k = 4) to group individual transactions into behavioural
clusters. An elbow / silhouette analysis is provided to aid k selection.

Typical clusters discovered from synthetic data:
| Cluster | Dominant system | Dominant counterparty | Pattern |
|---|---|---|---|
| 0 | INTERNAL | BANK | Mixed intraday flows |
| 1 | TARGET2 | CORPORATE | Pure outflows |
| 2 | TARGET2 | CORPORATE | Pure inflows |
| 3 | TARGET2 | BANK | Interbank outflows |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python data/generate_data.py

# 3. Run the full pipeline (from repo root)
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from src.data_processing import load_transactions, aggregate_to_buckets, calculate_liquidity_position
from src.feature_engineering import build_bucket_features, build_transaction_features, ANOMALY_FEATURE_COLS, CLUSTER_FEATURE_COLS
from src.anomaly_detection import detect_anomalies
from src.clustering import cluster_transactions, summarise_clusters
from src.visualization import plot_liquidity_position, plot_anomalies, plot_clusters

df  = load_transactions("data/raw/transactions.csv")
agg = aggregate_to_buckets(df)
liq = calculate_liquidity_position(agg)

bucket_feats  = build_bucket_features(liq)
tx_feats      = build_transaction_features(df)
anomaly_df    = detect_anomalies(bucket_feats, ANOMALY_FEATURE_COLS)
clustered     = cluster_transactions(df, tx_feats, CLUSTER_FEATURE_COLS)

plot_liquidity_position(liq)
plot_anomalies(anomaly_df)
plot_clusters(clustered)
print("Done – figures saved to figures/")
EOF

# 4. Or open the notebook
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## Expected Outputs

| Output | Location |
|---|---|
| Intraday liquidity position chart | `figures/liquidity_position.png` |
| Anomaly detection overlay | `figures/anomaly_detection.png` |
| Payment cluster scatter + volume | `figures/payment_clusters.png` |
| Elbow / silhouette plot | `figures/elbow_plot.png` |
| Processed liquidity CSV | `data/processed/liquidity_position.csv` |
| Anomaly results CSV | `data/processed/anomaly_results.csv` |
| Clustered transactions CSV | `data/processed/clustered_transactions.csv` |
| Cluster summary CSV | `data/processed/cluster_summary.csv` |

---

## Dependencies

See `requirements.txt`. Core libraries:

- **pandas ≥ 2.0** – data manipulation
- **numpy ≥ 1.24** – numerical operations
- **scikit-learn ≥ 1.3** – Isolation Forest, K-Means
- **matplotlib ≥ 3.7** – visualisation
- **jupyter / notebook** – interactive analysis

