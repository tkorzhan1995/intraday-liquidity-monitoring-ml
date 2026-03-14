"""
feature_engineering.py
-----------------------
Build ML-ready features from preprocessed intraday payment data.

Typical usage
-------------
    from src.feature_engineering import build_features

    df_features = build_features(df_processed)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature matrix from preprocessed transaction data.

    Features are computed at the **bank × day × hour** granularity and then
    merged back onto the transaction-level DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed transactions from :func:`src.data_processing.preprocess`.

    Returns
    -------
    pd.DataFrame
        Transaction-level DataFrame enriched with engineered features.
    """
    df = df.copy()

    df = _add_rolling_liquidity(df)
    df = _add_peak_hour_flag(df)
    df = _add_velocity_features(df)
    df = _add_net_position_features(df)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Individual feature builders
# ---------------------------------------------------------------------------

def _add_rolling_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 1-hour and 4-hour net flow per bank."""
    df = df.sort_values(["bank_id", "timestamp"]).copy()

    for window_hours in [1, 4]:
        window = f"{window_hours}h"
        col_name = f"rolling_net_{window_hours}h"
        results = []
        for _, group in df.groupby("bank_id"):
            group_indexed = group.set_index("timestamp")
            rolled = (
                group_indexed["signed_amount"]
                .rolling(window)
                .sum()
                .reset_index(drop=True)
            )
            rolled.index = group.index
            results.append(rolled)
        df[col_name] = pd.concat(results).sort_index()

    return df


def _add_peak_hour_flag(
    df: pd.DataFrame,
    peak_start: int = 9,
    peak_end: int = 12,
) -> pd.DataFrame:
    """Flag transactions occurring during morning peak hours."""
    df["is_peak_hour"] = (
        (df["hour"] >= peak_start) & (df["hour"] < peak_end)
    ).astype(int)
    return df


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count and total volume per bank per hour."""
    hourly = (
        df.groupby(["bank_id", "date", "hour"])
        .agg(
            txn_count_hourly=("transaction_id", "count"),
            volume_hourly=("amount", "sum"),
        )
        .reset_index()
    )
    df = df.merge(hourly, on=["bank_id", "date", "hour"], how="left")
    return df


def _add_net_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative net position per bank per day up to each transaction."""
    df = df.sort_values(["bank_id", "date", "timestamp"]).copy()
    df["cumulative_net_position"] = df.groupby(["bank_id", "date"])[
        "signed_amount"
    ].cumsum()
    return df


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "minute_of_day",
    "rolling_net_1h",
    "rolling_net_4h",
    "is_peak_hour",
    "txn_count_hourly",
    "volume_hourly",
    "cumulative_net_position",
]

# The last feature in NUMERIC_FEATURES is also commonly used as the
# supervised regression target (end-of-day net position).  When building
# a feature matrix for model training, exclude it from X and use it as y:
#
#   X, names = get_feature_matrix(df)   # does NOT include target
#   y = df["cumulative_net_position"]
TARGET_COLUMN = "cumulative_net_position"
_INPUT_FEATURES = [f for f in NUMERIC_FEATURES if f != TARGET_COLUMN]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a numeric feature matrix (excluding the target) and feature names.

    The target column (``cumulative_net_position``) is intentionally omitted
    from the returned matrix.  Use ``df[TARGET_COLUMN]`` as the label vector.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`build_features`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (X, feature_names) where X contains only the numeric input feature columns.
    """
    available = [col for col in _INPUT_FEATURES if col in df.columns]
    X = df[available].fillna(0)
    return X, available
