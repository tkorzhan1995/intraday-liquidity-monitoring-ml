import pandas as pd


def load_data(path):
    """Load payment transaction data from a CSV file.

    The CSV must contain a 'timestamp' column which will be parsed as datetime.

    Args:
        path: Path to the CSV file.

    Returns:
        pd.DataFrame with the 'timestamp' column as datetime64.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the CSV does not contain a 'timestamp' column.
        ValueError: If 'timestamp' values cannot be parsed as datetimes.
    """
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise KeyError("CSV must contain a 'timestamp' column.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_time_buckets(df):
    """Add a 'time_bucket' column by flooring timestamps to 5-minute intervals.

    Args:
        df: pd.DataFrame containing a datetime 'timestamp' column.

    Returns:
        A new pd.DataFrame with an additional 'time_bucket' column.

    Raises:
        KeyError: If 'timestamp' column is missing from the DataFrame.
    """
    if 'timestamp' not in df.columns:
        raise KeyError("DataFrame must contain a 'timestamp' column.")
    df = df.copy()
    df['time_bucket'] = df['timestamp'].dt.floor('5min')
    return df
