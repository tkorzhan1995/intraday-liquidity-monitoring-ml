from sklearn.ensemble import IsolationForest


def train_anomaly_model(data):
    """Train an Isolation Forest model to detect anomalies in liquidity data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a ``liquidity_change`` column with numeric values.

    Returns
    -------
    sklearn.ensemble.IsolationForest
        Fitted Isolation Forest model.
    """
    model = IsolationForest(
        contamination=0.01,
        random_state=42
    )

    model.fit(data[['liquidity_change']])

    return model


def detect_anomalies(model, data):
    """Predict anomalies in liquidity data using a fitted model.

    Adds an ``anomaly`` column to a copy of the input DataFrame where
    ``-1`` indicates an anomaly (unexpected liquidity drop) and ``1``
    indicates a normal observation.

    Parameters
    ----------
    model : sklearn.ensemble.IsolationForest
        A fitted Isolation Forest model.
    data : pandas.DataFrame
        DataFrame containing a ``liquidity_change`` column with numeric values.

    Returns
    -------
    pandas.DataFrame
        A copy of ``data`` with an additional ``anomaly`` column.
    """
    data = data.copy()
    data['anomaly'] = model.predict(data[['liquidity_change']])

    return data
