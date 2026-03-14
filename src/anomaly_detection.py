"""
anomaly_detection.py
--------------------
Detect unusual intraday liquidity events using unsupervised and
semi-supervised anomaly detection algorithms.

Typical usage
-------------
    from src.anomaly_detection import AnomalyDetector

    detector = AnomalyDetector(method="isolation_forest")
    detector.fit(X_train)
    labels = detector.predict(X)          # -1 = anomaly, 1 = normal
    scores = detector.decision_scores(X)  # lower = more anomalous
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

ANOMALY_MODEL_PATH = Path("models/anomaly_detector.pkl")
CONTAMINATION_DEFAULT = 0.02  # expected fraction of anomalies


# ---------------------------------------------------------------------------
# AnomalyDetector class
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Unified interface for unsupervised anomaly detection.

    Parameters
    ----------
    method : {'isolation_forest', 'lof', 'ocsvm'}
        Detection algorithm to use.
    contamination : float
        Expected proportion of anomalies in the dataset.
    random_state : int
        Random seed (used where applicable).
    """

    _SUPPORTED_METHODS = {"isolation_forest", "lof", "ocsvm"}

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = CONTAMINATION_DEFAULT,
        random_state: int = 42,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {self._SUPPORTED_METHODS}, got {method!r}"
            )
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._detector = self._build_detector()
        self._fitted = False

    # ------------------------------------------------------------------
    # Build underlying detector
    # ------------------------------------------------------------------

    def _build_detector(self):
        if self.method == "isolation_forest":
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.method == "lof":
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )
        # ocsvm
        return OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame | np.ndarray) -> "AnomalyDetector":
        """Fit the detector on normal (or mixed) transaction features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X_scaled = self._scaler.fit_transform(X)
        self._detector.fit(X_scaled)
        self._fitted = True
        logger.info(
            "AnomalyDetector (%s) fitted on %d samples", self.method, len(X_scaled)
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return +1 (normal) or -1 (anomaly) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in {-1, 1}.
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._detector.predict(X_scaled)

    def decision_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return raw anomaly scores (lower → more anomalous).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        if self.method == "lof":
            return self._detector.score_samples(X_scaled)
        return self._detector.decision_function(X_scaled)

    # ------------------------------------------------------------------
    # High-level analysis helpers
    # ------------------------------------------------------------------

    def label_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Append *anomaly_label* and *anomaly_score* columns to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction DataFrame with *feature_cols* present.
        feature_cols : list[str]
            Columns to use as input features.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with two extra columns.
        """
        df = df.copy()
        X = df[feature_cols].fillna(0)
        df["anomaly_label"] = self.predict(X)
        df["anomaly_score"] = self.decision_scores(X)
        return df

    def anomaly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarise anomalies by bank and date.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`label_dataframe`.

        Returns
        -------
        pd.DataFrame with columns: bank_id, date, n_anomalies, total_anomalous_amount.
        """
        anomalies = df[df["anomaly_label"] == -1]
        summary = (
            anomalies.groupby(["bank_id", "date"])
            .agg(
                n_anomalies=("anomaly_label", "count"),
                total_anomalous_amount=("amount", "sum"),
            )
            .reset_index()
            .sort_values("n_anomalies", ascending=False)
        )
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = ANOMALY_MODEL_PATH) -> None:
        """Save scaler + detector to a pickle file."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "detector": self._detector}, path)
        logger.info("AnomalyDetector saved to %s", path)

    @classmethod
    def load(cls, path: str | Path = ANOMALY_MODEL_PATH) -> "AnomalyDetector":
        """Load a previously saved AnomalyDetector.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        AnomalyDetector instance with _fitted = True.
        """
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance._scaler = payload["scaler"]
        instance._detector = payload["detector"]
        instance._fitted = True
        logger.info("AnomalyDetector loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detector not fitted yet. Call fit() first.")
