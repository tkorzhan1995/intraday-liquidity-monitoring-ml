"""
data_processing.py
------------------
Load, validate, and preprocess raw intraday payment transaction data.

Typical usage
-------------
    from src.data_processing import load_raw_data, preprocess

    df_raw = load_raw_data("data/raw/synthetic_intraday_payments_500k.csv")
    df = preprocess(df_raw)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/synthetic_intraday_payments_500k.csv")
PROCESSED_DATA_PATH = Path("data/processed/payments_processed.parquet")

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "bank_id",
    "transaction_type",
    "amount",
    "currency",
    "counterparty_id",
]

TRANSACTION_TYPES = ["CREDIT", "DEBIT"]
CURRENCIES = ["USD", "EUR", "GBP"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV file and return a DataFrame with typed columns.

    Parameters
    ----------
    path : str or Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw payment records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. "
            "Run generate_synthetic_data() first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("Loaded %d records from %s", len(df), path)
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw transaction data.

    Steps
    -----
    1. Drop duplicate transaction IDs.
    2. Remove rows with null amounts or timestamps.
    3. Parse and localise the timestamp to UTC.
    4. Add derived time columns (date, hour, minute_of_day).
    5. Normalise transaction_type to upper case.
    6. Add a signed_amount column (positive for CREDIT, negative for DEBIT).

    Parameters
    ----------
    df : pd.DataFrame
        Raw payment records as returned by :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched DataFrame.
    """
    df = df.copy()

    # 1. Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"])
    logger.info("Dropped %d duplicate records", before - len(df))

    # 2. Drop nulls in critical columns
    df = df.dropna(subset=["timestamp", "amount"])

    # 3. Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 4. Derived time features
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    # 5. Normalise transaction_type
    df["transaction_type"] = df["transaction_type"].str.upper()

    # 6. Signed amount
    df["signed_amount"] = np.where(
        df["transaction_type"] == "CREDIT",
        df["amount"],
        -df["amount"],
    )

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_records: int = 500_000,
    output_path: str | Path = RAW_DATA_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic intraday payment transactions dataset and save it.

    Parameters
    ----------
    n_records : int
        Number of transaction records to generate.
    output_path : str or Path
        Destination path for the CSV file.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The generated DataFrame (also written to *output_path*).
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_banks = 20
    n_counterparties = 100
    business_days = pd.bdate_range("2024-01-01", periods=250)

    # Random date + intraday time (business hours 08:00–18:00)
    day_indices = rng.integers(0, len(business_days), size=n_records)
    seconds_offset = rng.integers(0, 10 * 3600, size=n_records)  # 10-hour window
    timestamps = (
        business_days[day_indices].to_pydatetime()
        + pd.to_timedelta(seconds_offset + 8 * 3600, unit="s")
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:08d}" for i in range(n_records)],
            "timestamp": timestamps,
            "bank_id": [f"BANK{rng.integers(1, n_banks + 1):02d}" for _ in range(n_records)],
            "transaction_type": rng.choice(TRANSACTION_TYPES, size=n_records),
            "amount": np.round(rng.exponential(scale=50_000, size=n_records), 2),
            "currency": rng.choice(CURRENCIES, size=n_records),
            "counterparty_id": [
                f"CP{rng.integers(1, n_counterparties + 1):03d}"
                for _ in range(n_records)
            ],
        }
    )

    df.to_csv(output_path, index=False)
    logger.info("Saved %d synthetic records to %s", n_records, output_path)
    return df


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, path: str | Path = PROCESSED_DATA_PATH) -> None:
    """Persist the processed DataFrame as a Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved processed data to %s", path)


def load_processed(path: str | Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed Parquet file."""
    return pd.read_parquet(path)
