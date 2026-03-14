import pandas as pd


def calculate_liquidity_change(df):
    """Calculate the signed liquidity change for each transaction.

    Inflows are positive and outflows are negative.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with at least 'amount' and 'direction' columns.
        'direction' should be either 'INFLOW' or 'OUTFLOW'.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional 'liquidity_change' column.
    """
    df['liquidity_change'] = df['amount'] * df['direction'].map({'INFLOW': 1, 'OUTFLOW': -1})

    return df


def aggregate_liquidity(df):
    """Aggregate liquidity changes by time bucket to produce the intraday liquidity curve.

    Groups transactions by 'time_bucket', sums the 'liquidity_change' values,
    and computes the cumulative liquidity position.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with 'time_bucket' and 'liquidity_change' columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'time_bucket', 'liquidity_change' (net per bucket),
        and 'liquidity_position' (cumulative sum) columns.
    """
    liquidity = df.groupby('time_bucket')['liquidity_change'].sum().reset_index()

    liquidity = liquidity.sort_values('time_bucket').reset_index(drop=True)

    liquidity['liquidity_position'] = liquidity['liquidity_change'].cumsum()

    return liquidity
