"""
Feature engineering module for intraday liquidity monitoring.

Builds ML-ready feature matrices from aggregated bucket-level data
and raw transaction records.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bucket-level features
# ---------------------------------------------------------------------------


def build_bucket_features(liquidity_df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature matrix from the liquidity position DataFrame.

    Parameters
    ----------
    liquidity_df : pd.DataFrame
        Output of :func:`~src.data_processing.calculate_liquidity_position`.

    Returns
    -------
    pd.DataFrame
        Feature matrix with time-based and flow-based features.
    """
    df = liquidity_df.copy().sort_values("bucket").reset_index(drop=True)

    # Time-of-day features
    df["hour"] = df["bucket"].dt.hour
    df["minute"] = df["bucket"].dt.minute
    df["time_of_day"] = df["hour"] + df["minute"] / 60.0

    # Rolling statistics (3-bucket window ~ 15 minutes)
    window = 3
    df["rolling_net_flow_mean"] = (
        df["net_flow"].rolling(window, min_periods=1).mean()
    )
    df["rolling_net_flow_std"] = (
        df["net_flow"].rolling(window, min_periods=1).std().fillna(0)
    )
    df["rolling_inflow_mean"] = (
        df["total_inflow"].rolling(window, min_periods=1).mean()
    )
    df["rolling_outflow_mean"] = (
        df["total_outflow"].rolling(window, min_periods=1).mean()
    )

    # Lag features
    df["net_flow_lag1"] = df["net_flow"].shift(1).fillna(0)
    df["net_flow_lag2"] = df["net_flow"].shift(2).fillna(0)
    df["liquidity_position_lag1"] = df["liquidity_position"].shift(1).fillna(
        df["liquidity_position"].iloc[0]
    )

    # Liquidity-specific features
    df["liquidity_change"] = df["liquidity_position"].diff().fillna(0)
    df["liquidity_change_pct"] = (
        df["liquidity_change"] / df["liquidity_position_lag1"].replace(0, np.nan)
    ).fillna(0)
    df["outflow_inflow_ratio"] = (
        df["total_outflow"] / df["total_inflow"].replace(0, np.nan)
    ).fillna(0)

    return df


# ---------------------------------------------------------------------------
# Transaction-level features for clustering
# ---------------------------------------------------------------------------


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features for individual transactions.

    Parameters
    ----------
    transactions : pd.DataFrame
        Cleaned transaction DataFrame from
        :func:`~src.data_processing.load_transactions`.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with the input index.
    """
    df = transactions.copy()

    features = pd.DataFrame(index=df.index)
    features["amount"] = df["amount"]
    features["log_amount"] = np.log1p(df["amount"])
    features["hour"] = df["timestamp"].dt.hour
    features["minute"] = df["timestamp"].dt.minute
    features["time_of_day"] = features["hour"] + features["minute"] / 60.0
    features["is_outflow"] = (df["direction"] == "OUTFLOW").astype(int)

    # Encode categorical variables
    features["payment_system_encoded"] = (
        df["payment_system"]
        .astype("category")
        .cat.codes
    )
    features["counterparty_encoded"] = (
        df["counterparty_type"]
        .astype("category")
        .cat.codes
    )

    return features


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------


ANOMALY_FEATURE_COLS = [
    "net_flow",
    "total_inflow",
    "total_outflow",
    "transaction_count",
    "liquidity_change",
    "liquidity_change_pct",
    "outflow_inflow_ratio",
    "rolling_net_flow_mean",
    "rolling_net_flow_std",
    "time_of_day",
]

CLUSTER_FEATURE_COLS = [
    "log_amount",
    "time_of_day",
    "is_outflow",
    "payment_system_encoded",
    "counterparty_encoded",
]


def select_anomaly_features(bucket_features: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of columns used for anomaly detection."""
    available = [c for c in ANOMALY_FEATURE_COLS if c in bucket_features.columns]
    return bucket_features[available]


def select_cluster_features(transaction_features: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of columns used for clustering."""
    available = [c for c in CLUSTER_FEATURE_COLS if c in transaction_features.columns]
    return transaction_features[available]
