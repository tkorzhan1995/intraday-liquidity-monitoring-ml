"""
Data processing module for intraday liquidity monitoring.

Handles loading, cleaning, and aggregating payment transaction data
into 5-minute time buckets, and computes the running intraday liquidity
position throughout the settlement day.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "payment_id",
    "timestamp",
    "settlement_time",
    "payment_system",
    "counterparty_type",
    "direction",
    "amount",
]

PAYMENT_SYSTEMS = ["TARGET2", "SWIFT", "INTERNAL", "SECURITIES"]
COUNTERPARTY_TYPES = ["BANK", "CORPORATE", "CENTRAL_BANK", "CCP"]
DIRECTIONS = ["INFLOW", "OUTFLOW"]

BUCKET_FREQ = "5min"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_transactions(filepath: str | Path) -> pd.DataFrame:
    """Load raw transaction data from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed timestamps and validated schema.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp", "settlement_time"])
    _validate_schema(df)
    df = _clean_transactions(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and normalise column types."""
    df = df.copy()
    df["direction"] = df["direction"].str.upper().str.strip()
    df["payment_system"] = df["payment_system"].str.upper().str.strip()
    df["counterparty_type"] = df["counterparty_type"].str.upper().str.strip()
    df = df[df["amount"] > 0]
    df = df[df["direction"].isin(DIRECTIONS)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_to_buckets(
    df: pd.DataFrame, freq: str = BUCKET_FREQ
) -> pd.DataFrame:
    """Aggregate transactions into fixed time buckets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.
    freq : str
        Pandas offset alias for the bucket width (default ``"5min"``).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns:
        ``bucket``, ``total_inflow``, ``total_outflow``,
        ``net_flow``, ``transaction_count``, ``avg_transaction_size``.
    """
    df = df.copy()
    df["bucket"] = df["timestamp"].dt.floor(freq)

    inflows = (
        df[df["direction"] == "INFLOW"]
        .groupby("bucket")["amount"]
        .sum()
        .rename("total_inflow")
    )
    outflows = (
        df[df["direction"] == "OUTFLOW"]
        .groupby("bucket")["amount"]
        .sum()
        .rename("total_outflow")
    )
    counts = df.groupby("bucket")["payment_id"].count().rename("transaction_count")
    avg_size = df.groupby("bucket")["amount"].mean().rename("avg_transaction_size")

    agg = (
        pd.concat([inflows, outflows, counts, avg_size], axis=1)
        .fillna(0)
        .reset_index()
    )
    agg["net_flow"] = agg["total_inflow"] - agg["total_outflow"]
    return agg


# ---------------------------------------------------------------------------
# Liquidity position
# ---------------------------------------------------------------------------


def calculate_liquidity_position(
    agg: pd.DataFrame, initial_balance: float = 100_000_000.0
) -> pd.DataFrame:
    """Compute the running intraday liquidity position.

    Parameters
    ----------
    agg : pd.DataFrame
        Output of :func:`aggregate_to_buckets`.
    initial_balance : float
        Opening balance at the start of the settlement day.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with columns:
        ``cumulative_inflow``, ``cumulative_outflow``,
        ``liquidity_position``, ``liquidity_utilisation_pct``.
    """
    df = agg.copy().sort_values("bucket").reset_index(drop=True)
    df["cumulative_inflow"] = df["total_inflow"].cumsum()
    df["cumulative_outflow"] = df["total_outflow"].cumsum()
    df["liquidity_position"] = initial_balance + df["net_flow"].cumsum()
    df["liquidity_utilisation_pct"] = (
        (initial_balance - df["liquidity_position"]) / initial_balance * 100
    ).clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Per-system breakdown
# ---------------------------------------------------------------------------


def aggregate_by_system(
    df: pd.DataFrame, freq: str = BUCKET_FREQ
) -> pd.DataFrame:
    """Aggregate net flows per payment system per time bucket.

    Returns
    -------
    pd.DataFrame
        Columns: ``bucket``, ``payment_system``, ``net_flow``,
        ``transaction_count``.
    """
    df = df.copy()
    df["bucket"] = df["timestamp"].dt.floor(freq)
    df["signed_amount"] = df["amount"] * df["direction"].map(
        {"INFLOW": 1, "OUTFLOW": -1}
    )
    grouped = (
        df.groupby(["bucket", "payment_system"])
        .agg(
            net_flow=("signed_amount", "sum"),
            transaction_count=("payment_id", "count"),
        )
        .reset_index()
    )
    return grouped


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_processed(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save a processed DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
