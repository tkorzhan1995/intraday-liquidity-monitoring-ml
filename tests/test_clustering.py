import pandas as pd
import pytest
from src.clustering import cluster_payments


def make_df(amounts):
    return pd.DataFrame({'amount': amounts})


def test_cluster_column_added():
    df = make_df([10, 100, 1000, 10000, 20, 200, 2000, 20000])
    result = cluster_payments(df)
    assert 'cluster' in result.columns


def test_cluster_count():
    df = make_df(list(range(1, 41)))
    result = cluster_payments(df)
    assert result['cluster'].nunique() == 4


def test_cluster_labels_are_integers():
    df = make_df([50, 500, 5000, 50000] * 5)
    result = cluster_payments(df)
    assert result['cluster'].dtype.kind == 'i'


def test_cluster_length_unchanged():
    amounts = [10, 20, 30, 100, 200, 300, 1000, 2000, 3000, 10000]
    df = make_df(amounts)
    result = cluster_payments(df)
    assert len(result) == len(amounts)


def test_cluster_labels_in_range():
    df = make_df(list(range(1, 41)))
    result = cluster_payments(df)
    assert set(result['cluster'].unique()).issubset({0, 1, 2, 3})


def test_original_df_not_mutated():
    df = make_df([10, 100, 1000, 10000] * 5)
    original_columns = list(df.columns)
    cluster_payments(df)
    assert list(df.columns) == original_columns
