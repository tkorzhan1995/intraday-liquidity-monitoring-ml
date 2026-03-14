"""
Synthetic dataset generator for intraday liquidity monitoring.

Generates a realistic transaction-level dataset for one settlement day
and saves it to ``data/raw/transactions.csv``.

Usage
-----
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUTPUT_FILE = RAW_DIR / "transactions.csv"

PAYMENT_SYSTEMS = ["TARGET2", "SWIFT", "INTERNAL", "SECURITIES"]
COUNTERPARTY_TYPES = ["BANK", "CORPORATE", "CENTRAL_BANK", "CCP"]
DIRECTIONS = ["INFLOW", "OUTFLOW"]

# Settlement day window: 07:00 – 18:00 (660 minutes)
SETTLEMENT_START_HOUR = 7
SETTLEMENT_END_HOUR = 18

RNG = np.random.default_rng(42)


def _generate_transactions(n: int = 2000) -> pd.DataFrame:
    """Create a synthetic payment transaction dataset.

    The dataset simulates realistic intraday payment patterns:
    * Higher outflow activity in the morning (08:00–11:00).
    * Securities settlements clustered around 09:00 and 15:00.
    * A stress period around 13:30–14:00 with large corporate outflows.
    * Normal intraday noise throughout.
    """
    settlement_minutes = (SETTLEMENT_END_HOUR - SETTLEMENT_START_HOUR) * 60
    base_date = pd.Timestamp("2024-01-15")

    payment_ids = [f"PAY-{i:06d}" for i in range(1, n + 1)]

    # --- timestamps: mixture of three log-normal activity modes ---
    # Background: uniform throughout the day
    bg_count = int(n * 0.60)
    bg_minutes = RNG.uniform(0, settlement_minutes, bg_count)

    # Morning burst (60–240 min = 08:00–11:00)
    am_count = int(n * 0.25)
    am_minutes = RNG.normal(120, 40, am_count).clip(0, settlement_minutes)

    # Afternoon stress (390–450 min = 13:30–14:30)
    pm_count = n - bg_count - am_count
    pm_minutes = RNG.normal(410, 20, pm_count).clip(0, settlement_minutes)

    all_minutes = np.concatenate([bg_minutes, am_minutes, pm_minutes])
    RNG.shuffle(all_minutes)

    timestamps = [
        base_date
        + pd.Timedelta(hours=SETTLEMENT_START_HOUR)
        + pd.Timedelta(minutes=float(m))
        for m in all_minutes
    ]
    settlement_times = [
        t + pd.Timedelta(minutes=float(RNG.uniform(1, 10))) for t in timestamps
    ]

    # --- payment system: securities more likely in morning/afternoon peaks ---
    systems = RNG.choice(
        PAYMENT_SYSTEMS,
        size=n,
        p=[0.35, 0.30, 0.20, 0.15],
    )

    # --- counterparty ---
    counterparties = RNG.choice(
        COUNTERPARTY_TYPES,
        size=n,
        p=[0.40, 0.35, 0.10, 0.15],
    )

    # --- direction: slight outflow bias during stress window ---
    direction = []
    for m in all_minutes:
        if 390 <= m <= 450:  # stress window → more outflows
            d = RNG.choice(DIRECTIONS, p=[0.30, 0.70])
        else:
            d = RNG.choice(DIRECTIONS, p=[0.52, 0.48])
        direction.append(d)

    # --- amounts: log-normal; securities settlements are larger ---
    amounts = []
    for sys, d, m in zip(systems, direction, all_minutes):
        if sys == "SECURITIES":
            amt = RNG.lognormal(mean=13.5, sigma=1.0)  # ~700 k
        elif d == "OUTFLOW" and 390 <= m <= 450:
            amt = RNG.lognormal(mean=13.0, sigma=1.2)  # stress: large outflows
        else:
            amt = RNG.lognormal(mean=11.5, sigma=1.3)  # normal: ~100 k
        amounts.append(round(float(amt), 2))

    df = pd.DataFrame(
        {
            "payment_id": payment_ids,
            "timestamp": timestamps,
            "settlement_time": settlement_times,
            "payment_system": systems,
            "counterparty_type": counterparties,
            "direction": direction,
            "amount": amounts,
        }
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_and_save(n: int = 2000, output_path: Path | None = None) -> Path:
    """Generate the dataset and write it to CSV.

    Parameters
    ----------
    n : int
        Number of transactions to generate.
    output_path : Path, optional
        Override the default output path.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    path = output_path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _generate_transactions(n)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} transactions to {path}")
    return path


if __name__ == "__main__":
    generate_and_save()
