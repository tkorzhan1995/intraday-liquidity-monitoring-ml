"""
clustering.py
-------------
Cluster banks or time-windows by their intraday liquidity behaviour.

Typical usage
-------------
    from src.clustering import LiquidityClustering

    clusterer = LiquidityClustering(n_clusters=5)
    labels = clusterer.fit_predict(X)
    clusterer.plot_clusters(X, labels, save_path="figures/clustering.png")
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CLUSTER_MODEL_PATH = Path("models/cluster_model.pkl")


# ---------------------------------------------------------------------------
# LiquidityClustering class
# ---------------------------------------------------------------------------

class LiquidityClustering:
    """Cluster banks / time-windows by intraday liquidity patterns.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    method : {'kmeans', 'agglomerative'}
        Clustering algorithm.
    random_state : int
        Random seed.
    """

    _SUPPORTED_METHODS = {"kmeans", "agglomerative"}

    def __init__(
        self,
        n_clusters: int = 5,
        method: str = "kmeans",
        random_state: int = 42,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {self._SUPPORTED_METHODS}, got {method!r}"
            )
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = self._build_model()
        self._fitted = False

    # ------------------------------------------------------------------
    # Build underlying model
    # ------------------------------------------------------------------

    def _build_model(self):
        if self.method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
        return AgglomerativeClustering(n_clusters=self.n_clusters)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame | np.ndarray) -> "LiquidityClustering":
        """Fit the clustering model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True
        logger.info(
            "LiquidityClustering (%s, k=%d) fitted on %d samples",
            self.method,
            self.n_clusters,
            len(X_scaled),
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Assign cluster labels to samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with integer cluster IDs.
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        if self.method == "agglomerative":
            # AgglomerativeClustering has no predict(); refit on new data
            return self._model.fit_predict(X_scaled)
        return self._model.predict(X_scaled)

    def fit_predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit and return cluster labels in one step."""
        self.fit(X)
        return self.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame | np.ndarray) -> dict[str, float]:
        """Compute clustering quality metrics.

        Returns
        -------
        dict with keys: silhouette_score, davies_bouldin_score, inertia (kmeans only)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        labels = self.predict(X)
        metrics: dict[str, float] = {
            "silhouette_score": silhouette_score(X_scaled, labels),
            "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        }
        if self.method == "kmeans":
            metrics["inertia"] = float(self._model.inertia_)
        logger.info("Clustering metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Elbow / silhouette search
    # ------------------------------------------------------------------

    @staticmethod
    def find_optimal_k(
        X: pd.DataFrame | np.ndarray,
        k_range: range = range(2, 11),
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute inertia and silhouette scores for a range of k values.

        Parameters
        ----------
        X : array-like
        k_range : range
            Range of cluster counts to evaluate.
        random_state : int

        Returns
        -------
        pd.DataFrame with columns: k, inertia, silhouette_score
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = km.fit_predict(X_scaled)
            rows.append(
                {
                    "k": k,
                    "inertia": km.inertia_,
                    "silhouette_score": silhouette_score(X_scaled, labels),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    def cluster_profiles(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "cluster",
    ) -> pd.DataFrame:
        """Compute per-cluster feature means for interpretability.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns and a cluster label column.
        feature_cols : list[str]
            Feature columns to aggregate.
        label_col : str
            Column containing the cluster assignments.

        Returns
        -------
        pd.DataFrame with cluster label as index and feature means as columns.
        """
        return df.groupby(label_col)[feature_cols].mean()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_clusters(
        self,
        X: pd.DataFrame | np.ndarray,
        labels: np.ndarray,
        save_path: str | Path | None = None,
        title: str = "Intraday Liquidity Clusters (PCA)",
    ) -> None:
        """2-D PCA scatter plot colour-coded by cluster.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        labels : array-like of shape (n_samples,)
            Cluster assignments.
        save_path : str, Path, or None
            If provided, save the figure to this path.
        title : str
            Plot title.
        """
        X_arr = np.array(X)
        pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(self._scaler.transform(X_arr))

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.6,
            s=10,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(title)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            logger.info("Cluster plot saved to %s", save_path)
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = CLUSTER_MODEL_PATH) -> None:
        """Save scaler + model to a pickle file."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "model": self._model}, path)
        logger.info("LiquidityClustering saved to %s", path)

    @classmethod
    def load(cls, path: str | Path = CLUSTER_MODEL_PATH) -> "LiquidityClustering":
        """Load a previously saved clustering model.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        LiquidityClustering instance.
        """
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance._scaler = payload["scaler"]
        instance._model = payload["model"]
        instance._fitted = True
        logger.info("LiquidityClustering loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
