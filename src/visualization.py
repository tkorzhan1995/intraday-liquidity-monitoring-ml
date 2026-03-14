"""
Visualization module for intraday liquidity monitoring.

Produces publication-ready matplotlib figures and saves them to the
``figures/`` directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def _ensure_figures_dir(output_dir: Path | None = None) -> Path:
    d = output_dir or FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1. Intraday liquidity position
# ---------------------------------------------------------------------------


def plot_liquidity_position(
    liquidity_df: pd.DataFrame,
    initial_balance: float = 100_000_000.0,
    output_dir: Path | None = None,
    filename: str = "liquidity_position.png",
) -> Path:
    """Plot the running intraday liquidity position.

    Parameters
    ----------
    liquidity_df : pd.DataFrame
        Output of :func:`~src.data_processing.calculate_liquidity_position`.
    initial_balance : float
        Opening balance – used to draw a reference line.
    output_dir : Path, optional
        Directory to save the figure (defaults to ``figures/``).
    filename : str
        Output file name.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top panel: liquidity position
    ax1 = axes[0]
    ax1.plot(
        liquidity_df["bucket"],
        liquidity_df["liquidity_position"] / 1e6,
        color="#1f77b4",
        linewidth=1.5,
        label="Liquidity position",
    )
    ax1.axhline(
        initial_balance / 1e6,
        color="grey",
        linestyle="--",
        linewidth=1,
        label="Opening balance",
    )
    ax1.set_ylabel("Position (M)", fontsize=11)
    ax1.set_title("Intraday Liquidity Position", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom panel: net flow per bucket
    ax2 = axes[1]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in liquidity_df["net_flow"]]
    ax2.bar(
        liquidity_df["bucket"],
        liquidity_df["net_flow"] / 1e6,
        color=colors,
        width=pd.Timedelta("4min"),
        label="Net flow per bucket",
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Net flow (M)", fontsize=11)
    ax2.set_xlabel("Time", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    out = _ensure_figures_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Anomaly overlay
# ---------------------------------------------------------------------------


def plot_anomalies(
    bucket_features: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "anomaly_detection.png",
) -> Path:
    """Overlay detected anomalies on the liquidity position chart.

    Parameters
    ----------
    bucket_features : pd.DataFrame
        Output of :func:`~src.anomaly_detection.detect_anomalies`
        (must contain ``bucket``, ``liquidity_position``, ``is_anomaly``,
        ``anomaly_score`` columns).
    output_dir : Path, optional
    filename : str

    Returns
    -------
    Path
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    normal = bucket_features[~bucket_features["is_anomaly"]]
    anomalies = bucket_features[bucket_features["is_anomaly"]]

    # Top panel: position with anomaly markers
    ax1 = axes[0]
    ax1.plot(
        bucket_features["bucket"],
        bucket_features["liquidity_position"] / 1e6,
        color="#1f77b4",
        linewidth=1.5,
        label="Liquidity position",
    )
    ax1.scatter(
        anomalies["bucket"],
        anomalies["liquidity_position"] / 1e6,
        color="#d62728",
        zorder=5,
        s=60,
        label=f"Anomaly ({len(anomalies)} detected)",
    )
    ax1.set_ylabel("Position (M)", fontsize=11)
    ax1.set_title(
        "Intraday Liquidity – Anomaly Detection (Isolation Forest)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom panel: anomaly scores
    ax2 = axes[1]
    ax2.plot(
        bucket_features["bucket"],
        bucket_features["anomaly_score"],
        color="#7f7f7f",
        linewidth=1,
        label="Anomaly score",
    )
    ax2.scatter(
        anomalies["bucket"],
        anomalies["anomaly_score"],
        color="#d62728",
        zorder=5,
        s=40,
    )
    ax2.set_ylabel("Isolation Forest score", fontsize=11)
    ax2.set_xlabel("Time", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.tight_layout()
    out = _ensure_figures_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Cluster visualisation
# ---------------------------------------------------------------------------


def plot_clusters(
    transactions_with_clusters: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "payment_clusters.png",
) -> Path:
    """Scatter-plot of payments coloured by cluster, with a summary table.

    Parameters
    ----------
    transactions_with_clusters : pd.DataFrame
        Output of :func:`~src.clustering.cluster_transactions`.
    output_dir : Path, optional
    filename : str

    Returns
    -------
    Path
    """
    df = transactions_with_clusters.copy()
    df["hour_float"] = (
        df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    )
    df["log_amount"] = np.log1p(df["amount"])

    n_clusters = df["cluster"].nunique()
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: scatter amount vs time
    ax1 = axes[0]
    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        ax1.scatter(
            subset["hour_float"],
            subset["log_amount"],
            alpha=0.4,
            s=15,
            color=cmap(cid),
            label=f"Cluster {cid}",
        )
    ax1.set_xlabel("Time of day (hours)", fontsize=11)
    ax1.set_ylabel("log(1 + amount)", fontsize=11)
    ax1.set_title("Payment Clusters – Amount vs Time", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Right: bar chart of total volume per cluster split by direction
    ax2 = axes[1]
    cluster_dir = (
        df.groupby(["cluster", "direction"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .div(1e6)
    )
    cluster_dir.plot(
        kind="bar",
        ax=ax2,
        color={"INFLOW": "#2ca02c", "OUTFLOW": "#d62728"},
        edgecolor="white",
    )
    ax2.set_xlabel("Cluster", fontsize=11)
    ax2.set_ylabel("Total amount (M)", fontsize=11)
    ax2.set_title("Cluster Volume by Direction", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="x", rotation=0)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = _ensure_figures_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Elbow / silhouette plot
# ---------------------------------------------------------------------------


def plot_elbow(
    cluster_scores: dict,
    output_dir: Path | None = None,
    filename: str = "elbow_plot.png",
) -> Path:
    """Plot inertia and silhouette score to aid k selection.

    Parameters
    ----------
    cluster_scores : dict
        Output of :func:`~src.clustering.find_optimal_clusters`.
    output_dir : Path, optional
    filename : str

    Returns
    -------
    Path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(cluster_scores["k"], cluster_scores["inertia"], marker="o", color="#1f77b4")
    ax1.set_xlabel("Number of clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia", fontsize=11)
    ax1.set_title("Elbow Curve", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.plot(
        cluster_scores["k"],
        cluster_scores["silhouette"],
        marker="o",
        color="#ff7f0e",
    )
    ax2.set_xlabel("Number of clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette score", fontsize=11)
    ax2.set_title("Silhouette Score vs k", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = _ensure_figures_dir(output_dir) / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
