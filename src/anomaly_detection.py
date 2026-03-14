"""
Anomaly detection module for intraday liquidity monitoring.

Uses Isolation Forest to identify unusual liquidity movements
in aggregated 5-minute bucket data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class LiquidityAnomalyDetector:
    """Isolation Forest-based anomaly detector for liquidity time-series.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the data (0 < contamination < 0.5).
    n_estimators : int
        Number of trees in the Isolation Forest.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "LiquidityAnomalyDetector":
        """Fit the scaler and Isolation Forest on the feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from
            :func:`~src.feature_engineering.select_anomaly_features`.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly labels: -1 for anomalies, 1 for normal.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as used in :meth:`fit`).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return the raw anomaly score for each sample.

        Lower (more negative) scores indicate more anomalous observations.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    # ------------------------------------------------------------------
    # Convenience helper
    # ------------------------------------------------------------------

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the model and return predictions in one call."""
        self.fit(X)
        return self.predict(X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted yet. Call .fit() or .fit_predict() first."
            )


# ---------------------------------------------------------------------------
# High-level pipeline helper
# ---------------------------------------------------------------------------


def detect_anomalies(
    bucket_features: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run anomaly detection and annotate the bucket feature DataFrame.

    Parameters
    ----------
    bucket_features : pd.DataFrame
        Full bucket-level feature DataFrame (output of
        :func:`~src.feature_engineering.build_bucket_features`).
    feature_cols : list[str]
        Column names to use as input features.
    contamination : float
        Expected anomaly proportion.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional columns:
        ``is_anomaly`` (bool) and ``anomaly_score`` (float).
    """
    df = bucket_features.copy()
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols]

    detector = LiquidityAnomalyDetector(
        contamination=contamination, random_state=random_state
    )
    labels = detector.fit_predict(X)
    scores = detector.anomaly_scores(X)

    df["is_anomaly"] = labels == -1
    df["anomaly_score"] = scores
    return df
