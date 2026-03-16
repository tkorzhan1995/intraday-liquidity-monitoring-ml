"""Intraday Liquidity Monitoring ML Pipeline.

This module provides utilities for generating synthetic intraday payment
transaction data, detecting liquidity anomalies using Isolation Forest, and
visualising the results.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend suitable for testing / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_transaction_data(
    seed: int = 42,
    n_buckets: int = 48,
    start_time: str = "08:00",
    freq: str = "30min",
) -> pd.DataFrame:
    """Generate synthetic intraday transaction data with a liquidity drop at 11:30.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    n_buckets:
        Number of 30-minute time buckets in the trading day.
    start_time:
        Start time of the first bucket (HH:MM string).
    freq:
        Pandas frequency string for the time buckets.

    Returns
    -------
    pd.DataFrame
        Columns: ``time_bucket`` (datetime), ``transaction_volume`` (float),
        ``net_flow`` (float), ``liquidity_change`` (float).
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range(start=f"2024-01-01 {start_time}", periods=n_buckets, freq=freq)

    # Typical intraday volume profile (U-shaped)
    base_volume = 1_000 + 500 * np.cos(np.linspace(0, np.pi, n_buckets)) ** 2
    transaction_volume = base_volume + rng.normal(0, 50, n_buckets)

    # Net flow with random noise
    net_flow = rng.normal(0, 100, n_buckets)

    # Liquidity change derived from net flow with added noise
    liquidity_change = net_flow + rng.normal(0, 20, n_buckets)

    # Inject a large liquidity drop at ~11:30 (bucket index 7 from 08:00)
    drop_index = int((pd.Timestamp(f"2024-01-01 11:30") - pd.Timestamp(f"2024-01-01 {start_time}"))
                     / pd.Timedelta(freq))
    if 0 <= drop_index < n_buckets:
        liquidity_change[drop_index] -= 600

    return pd.DataFrame(
        {
            "time_bucket": times,
            "transaction_volume": transaction_volume,
            "net_flow": net_flow,
            "liquidity_change": liquidity_change,
        }
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    data: pd.DataFrame,
    features: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Detect anomalies in the transaction data using Isolation Forest.

    Parameters
    ----------
    data:
        DataFrame produced by :func:`generate_transaction_data` (or any
        DataFrame with numeric feature columns).
    features:
        List of column names to use as model features.  Defaults to
        ``["transaction_volume", "net_flow", "liquidity_change"]``.
    contamination:
        Expected proportion of outliers in the dataset.
    random_state:
        Random seed for the Isolation Forest.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``anomaly`` column where ``-1``
        indicates an anomaly and ``1`` indicates a normal observation.
    """
    if features is None:
        features = ["transaction_volume", "net_flow", "liquidity_change"]

    model = IsolationForest(contamination=contamination, random_state=random_state)
    result = data.copy()
    result["anomaly"] = model.fit_predict(data[features])
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_anomalies(
    data: pd.DataFrame,
    time_col: str = "time_bucket",
    value_col: str = "liquidity_change",
    anomaly_col: str = "anomaly",
    title: str = "Liquidity Anomaly Detection",
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot the liquidity time-series and highlight detected anomalies.

    Normal observations are drawn as a continuous line; anomalous points are
    overlaid as red scatter markers so that unusual liquidity movements
    (e.g. the 11:30 drop) are immediately visible.

    Parameters
    ----------
    data:
        DataFrame containing at least *time_col*, *value_col*, and
        *anomaly_col* columns.  The ``anomaly`` column must follow the
        sklearn convention where ``-1`` marks anomalies and ``1`` marks
        normal points.
    time_col:
        Name of the column with time-bucket values (x-axis).
    value_col:
        Name of the column with the liquidity metric (y-axis).
    anomaly_col:
        Name of the column containing anomaly labels (``-1`` / ``1``).
    title:
        Title displayed at the top of the chart.
    figsize:
        ``(width, height)`` of the figure in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure, which the caller can save or display.
    """
    fig = plt.figure(figsize=figsize)

    # Full liquidity-change time series
    plt.plot(data[time_col], data[value_col], label="Liquidity Change", color="steelblue")

    # Anomalous points highlighted in red
    anomalies = data[data[anomaly_col] == -1]
    plt.scatter(
        anomalies[time_col],
        anomalies[value_col],
        color="red",
        zorder=5,
        label="Anomaly",
    )

    plt.title(title)
    plt.xlabel("Time Bucket")
    plt.ylabel("Liquidity Change")
    plt.legend()
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(output_path: str | None = None) -> pd.DataFrame:
    """Run the full monitoring pipeline and display/save the anomaly chart.

    Parameters
    ----------
    output_path:
        If provided, the figure is saved to this path instead of being shown
        interactively.

    Returns
    -------
    pd.DataFrame
        The final dataset including the ``anomaly`` column.
    """
    data = generate_transaction_data()
    data = detect_anomalies(data)

    fig = plot_anomalies(data)

    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()

    plt.close(fig)
    return data


if __name__ == "__main__":
    run_pipeline()
