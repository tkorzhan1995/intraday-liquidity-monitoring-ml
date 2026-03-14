"""
liquidity_model.py
------------------
Train, evaluate, and persist a supervised model that predicts the end-of-day
net liquidity position for a bank given its intraday transaction patterns.

Typical usage
-------------
    from src.liquidity_model import LiquidityModel

    model = LiquidityModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model.save("models/trained_model.pkl")
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/trained_model.pkl")


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class LiquidityModel:
    """Gradient-boosted regressor for end-of-day net liquidity prediction.

    Parameters
    ----------
    model_type : {'gbm', 'rf'}
        Underlying estimator: Gradient Boosting Machine or Random Forest.
    n_estimators : int
        Number of boosting stages / trees.
    max_depth : int
        Maximum depth of individual trees.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_type: str = "gbm",
        n_estimators: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LiquidityModel":
        """Fit the model pipeline on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target: end-of-day net position.

        Returns
        -------
        self
        """
        estimator = self._build_estimator()
        self._pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", estimator)]
        )
        self._pipeline.fit(X, y)
        logger.info("LiquidityModel fitted on %d samples", len(y))
        return self

    def _build_estimator(self):
        if self.model_type == "gbm":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown model_type: {self.model_type!r}. Use 'gbm' or 'rf'.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted end-of-day net liquidity positions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return self._pipeline.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Compute regression metrics on a held-out set.

        Returns
        -------
        dict with keys: mae, rmse, r2
        """
        self._check_fitted()
        y_pred = self.predict(X)
        metrics = {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": r2_score(y, y_pred),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
    ) -> dict[str, float]:
        """Run k-fold cross-validation and return mean ± std scores.

        Returns
        -------
        dict with keys: cv_mae_mean, cv_mae_std
        """
        estimator = self._build_estimator()
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        result = {
            "cv_mae_mean": float(-scores.mean()),
            "cv_mae_std": float(scores.std()),
        }
        logger.info("Cross-validation results: %s", result)
        return result

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self, feature_names: list[str]) -> pd.Series:
        """Return a Series of feature importances sorted descending.

        Parameters
        ----------
        feature_names : list[str]
            Names corresponding to columns in the training feature matrix.

        Returns
        -------
        pd.Series indexed by feature name.
        """
        self._check_fitted()
        model = self._pipeline.named_steps["model"]
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = MODEL_PATH) -> None:
        """Serialise the fitted pipeline to a pickle file."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "LiquidityModel":
        """Deserialise a previously saved pipeline.

        Parameters
        ----------
        path : str or Path
            Location of the saved ``.pkl`` file.

        Returns
        -------
        LiquidityModel with _pipeline populated.
        """
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def train_and_save(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "cumulative_net_position",
    test_size: float = 0.2,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, float]:
    """End-to-end helper: split, train, evaluate, and save.

    Parameters
    ----------
    df_features : pd.DataFrame
        Output of :func:`src.feature_engineering.build_features`.
    feature_cols : list[str]
        Column names to use as features.
    target_col : str
        Column to predict.
    test_size : float
        Fraction of data held out for evaluation.
    model_path : str or Path
        Where to save the trained model.

    Returns
    -------
    dict with evaluation metrics from the test set.
    """
    X = df_features[feature_cols].fillna(0)
    y = df_features[target_col].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LiquidityModel()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    model.save(model_path)
    return metrics
