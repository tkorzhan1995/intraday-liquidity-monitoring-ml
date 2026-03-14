"""
Clustering module for intraday liquidity monitoring.

Uses K-Means to identify groups of payments responsible for
distinct liquidity events (e.g. corporate outflows, CCP settlements).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class PaymentClusterer:
    """K-Means clustering of individual payment transactions.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.
    """

    # Human-readable labels assigned after profiling each cluster
    CLUSTER_LABEL_MAP: dict[int, str] = {}

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "PaymentClusterer":
        """Fit the scaler and K-Means on the feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Transaction-level feature matrix from
            :func:`~src.feature_engineering.select_cluster_features`.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign cluster labels to transactions.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,) with integer cluster ids.
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the model and return cluster labels in one call."""
        self.fit(X)
        return self.predict(X)

    def silhouette(self, X: pd.DataFrame) -> float:
        """Return the silhouette score for the fitted clustering."""
        self._check_fitted()
        labels = self.predict(X)
        X_scaled = self.scaler.transform(X)
        return float(silhouette_score(X_scaled, labels))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted yet. Call .fit() or .fit_predict() first."
            )


# ---------------------------------------------------------------------------
# Optimal k selection
# ---------------------------------------------------------------------------


def find_optimal_clusters(
    X: pd.DataFrame,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> dict[str, list]:
    """Compute inertia and silhouette scores for a range of k values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (same format as used by :class:`PaymentClusterer`).
    k_range : range
        Values of k to evaluate.
    random_state : int

    Returns
    -------
    dict with keys ``"k"``, ``"inertia"``, ``"silhouette"``.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results: dict[str, list] = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0
        results["silhouette"].append(float(sil))

    return results


# ---------------------------------------------------------------------------
# High-level pipeline helper
# ---------------------------------------------------------------------------


def cluster_transactions(
    transactions: pd.DataFrame,
    transaction_features: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run K-Means clustering and annotate the transactions DataFrame.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw / cleaned transaction DataFrame.
    transaction_features : pd.DataFrame
        Feature matrix aligned with ``transactions`` index.
    feature_cols : list[str]
        Column names to use from ``transaction_features``.
    n_clusters : int
    random_state : int

    Returns
    -------
    pd.DataFrame
        ``transactions`` with an additional ``cluster`` column.
    """
    available_cols = [c for c in feature_cols if c in transaction_features.columns]
    X = transaction_features[available_cols]

    clusterer = PaymentClusterer(n_clusters=n_clusters, random_state=random_state)
    labels = clusterer.fit_predict(X)

    result = transactions.copy()
    result["cluster"] = labels
    return result


def summarise_clusters(
    transactions_with_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Produce a per-cluster summary table.

    Parameters
    ----------
    transactions_with_clusters : pd.DataFrame
        Output of :func:`cluster_transactions`.

    Returns
    -------
    pd.DataFrame
        Summary with cluster-level statistics.
    """
    df = transactions_with_clusters.copy()
    df["signed_amount"] = df["amount"] * df["direction"].map(
        {"INFLOW": 1, "OUTFLOW": -1}
    )

    summary = (
        df.groupby("cluster")
        .agg(
            transaction_count=("payment_id", "count"),
            total_amount=("amount", "sum"),
            mean_amount=("amount", "mean"),
            net_flow=("signed_amount", "sum"),
            dominant_system=("payment_system", lambda s: s.value_counts().idxmax()),
            dominant_counterparty=(
                "counterparty_type",
                lambda s: s.value_counts().idxmax(),
            ),
            pct_outflow=(
                "direction",
                lambda s: (s == "OUTFLOW").mean() * 100,
            ),
        )
        .reset_index()
    )
    return summary
