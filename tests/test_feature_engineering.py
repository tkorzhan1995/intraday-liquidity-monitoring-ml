import pandas as pd
import pytest
from src.feature_engineering import calculate_liquidity_change, aggregate_liquidity


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        'time_bucket': ['09:00', '09:00', '09:30', '09:30', '10:00'],
        'amount': [100, 50, 200, 80, 150],
        'direction': ['INFLOW', 'OUTFLOW', 'INFLOW', 'OUTFLOW', 'INFLOW'],
    })


def test_calculate_liquidity_change_inflow(sample_transactions):
    result = calculate_liquidity_change(sample_transactions.copy())
    assert result.loc[0, 'liquidity_change'] == 100


def test_calculate_liquidity_change_outflow(sample_transactions):
    result = calculate_liquidity_change(sample_transactions.copy())
    assert result.loc[1, 'liquidity_change'] == -50


def test_calculate_liquidity_change_all_rows(sample_transactions):
    result = calculate_liquidity_change(sample_transactions.copy())
    expected = [100, -50, 200, -80, 150]
    assert list(result['liquidity_change']) == expected


def test_calculate_liquidity_change_adds_column(sample_transactions):
    result = calculate_liquidity_change(sample_transactions.copy())
    assert 'liquidity_change' in result.columns


def test_aggregate_liquidity_net_per_bucket(sample_transactions):
    df = calculate_liquidity_change(sample_transactions.copy())
    result = aggregate_liquidity(df)
    net = result.set_index('time_bucket')['liquidity_change']
    assert net['09:00'] == 50    # 100 - 50
    assert net['09:30'] == 120   # 200 - 80
    assert net['10:00'] == 150


def test_aggregate_liquidity_cumulative_position(sample_transactions):
    df = calculate_liquidity_change(sample_transactions.copy())
    result = aggregate_liquidity(df)
    expected_positions = [50, 170, 320]  # cumsum of [50, 120, 150]
    assert list(result['liquidity_position']) == expected_positions


def test_aggregate_liquidity_columns(sample_transactions):
    df = calculate_liquidity_change(sample_transactions.copy())
    result = aggregate_liquidity(df)
    assert 'time_bucket' in result.columns
    assert 'liquidity_change' in result.columns
    assert 'liquidity_position' in result.columns


def test_aggregate_liquidity_sorted_by_time_bucket():
    df = pd.DataFrame({
        'time_bucket': ['10:00', '09:00', '09:30'],
        'liquidity_change': [150, 50, 120],
    })
    result = aggregate_liquidity(df)
    assert list(result['time_bucket']) == ['09:00', '09:30', '10:00']
    assert list(result['liquidity_position']) == [50, 170, 320]
