from sklearn.cluster import KMeans


def cluster_payments(df):
    """Cluster payment transactions using K-Means.

    Assigns each transaction to one of 4 clusters based on amount,
    helping identify categories such as securities settlements,
    corporate payments, and retail transactions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least an ``amount`` column.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional ``cluster`` column
        containing integer cluster labels (0–3).
    """
    df = df.copy()
    features = df[['amount']]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)
    return df
