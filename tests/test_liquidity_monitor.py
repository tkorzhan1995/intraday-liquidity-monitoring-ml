"""Tests for the intraday liquidity monitoring ML pipeline."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from liquidity_monitor import (
    detect_anomalies,
    generate_transaction_data,
    plot_anomalies,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# generate_transaction_data
# ---------------------------------------------------------------------------

class TestGenerateTransactionData:
    def test_returns_dataframe(self):
        df = generate_transaction_data()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = generate_transaction_data()
        assert set(df.columns) == {"time_bucket", "transaction_volume", "net_flow", "liquidity_change"}

    def test_default_row_count(self):
        df = generate_transaction_data()
        assert len(df) == 48

    def test_custom_n_buckets(self):
        df = generate_transaction_data(n_buckets=24)
        assert len(df) == 24

    def test_time_bucket_dtype(self):
        df = generate_transaction_data()
        assert pd.api.types.is_datetime64_any_dtype(df["time_bucket"])

    def test_reproducible_with_same_seed(self):
        df1 = generate_transaction_data(seed=0)
        df2 = generate_transaction_data(seed=0)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_transaction_data(seed=0)
        df2 = generate_transaction_data(seed=99)
        assert not df1["liquidity_change"].equals(df2["liquidity_change"])

    def test_liquidity_drop_at_11_30(self):
        """The injected drop at 11:30 should be the minimum liquidity change."""
        df = generate_transaction_data()
        min_idx = df["liquidity_change"].idxmin()
        bucket_time = df.loc[min_idx, "time_bucket"]
        assert bucket_time.hour == 11
        assert bucket_time.minute == 30


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

class TestDetectAnomalies:
    def setup_method(self):
        self.data = generate_transaction_data()

    def test_adds_anomaly_column(self):
        result = detect_anomalies(self.data)
        assert "anomaly" in result.columns

    def test_anomaly_labels_are_valid(self):
        result = detect_anomalies(self.data)
        assert set(result["anomaly"].unique()).issubset({-1, 1})

    def test_does_not_modify_original(self):
        original_cols = list(self.data.columns)
        detect_anomalies(self.data)
        assert list(self.data.columns) == original_cols

    def test_detects_11_30_drop_as_anomaly(self):
        """The large liquidity drop injected at 11:30 must be flagged."""
        result = detect_anomalies(self.data)
        drop_row = result[
            (result["time_bucket"].dt.hour == 11)
            & (result["time_bucket"].dt.minute == 30)
        ]
        assert not drop_row.empty
        assert (drop_row["anomaly"] == -1).all()

    def test_custom_features(self):
        result = detect_anomalies(self.data, features=["liquidity_change"])
        assert "anomaly" in result.columns

    def test_returns_same_length(self):
        result = detect_anomalies(self.data)
        assert len(result) == len(self.data)


# ---------------------------------------------------------------------------
# plot_anomalies
# ---------------------------------------------------------------------------

class TestPlotAnomalies:
    def setup_method(self):
        raw = generate_transaction_data()
        self.data = detect_anomalies(raw)

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fig = plot_anomalies(self.data)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_one_axes(self):
        fig = plot_anomalies(self.data)
        assert len(fig.axes) == 1

    def test_title_is_set(self):
        fig = plot_anomalies(self.data)
        assert fig.axes[0].get_title() == "Liquidity Anomaly Detection"

    def test_custom_title(self):
        fig = plot_anomalies(self.data, title="My Custom Title")
        assert fig.axes[0].get_title() == "My Custom Title"

    def test_line_is_plotted(self):
        fig = plot_anomalies(self.data)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1, "Expected at least one Line2D (normal data)"

    def test_scatter_is_plotted_for_anomalies(self):
        """Anomaly scatter should contain exactly the anomalous rows."""
        fig = plot_anomalies(self.data)
        ax = fig.axes[0]
        # PathCollections are scatter plots
        path_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        assert len(path_collections) >= 1, "Expected at least one scatter (anomalies)"

        scatter = path_collections[0]
        offsets = scatter.get_offsets()
        n_anomalies = (self.data["anomaly"] == -1).sum()
        assert len(offsets) == n_anomalies

    def test_no_anomalies_empty_scatter(self):
        """When there are no anomalies the scatter should be empty."""
        data_no_anomaly = self.data.copy()
        data_no_anomaly["anomaly"] = 1  # all normal
        fig = plot_anomalies(data_no_anomaly)
        ax = fig.axes[0]
        path_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        if path_collections:
            assert len(path_collections[0].get_offsets()) == 0

    def test_legend_present(self):
        fig = plot_anomalies(self.data)
        legend = fig.axes[0].get_legend()
        assert legend is not None

    def test_custom_figsize(self):
        fig = plot_anomalies(self.data, figsize=(8, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8)
        assert h == pytest.approx(4)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def teardown_method(self):
        plt.close("all")

    def test_returns_dataframe(self):
        result = run_pipeline()
        assert isinstance(result, pd.DataFrame)

    def test_result_has_anomaly_column(self):
        result = run_pipeline()
        assert "anomaly" in result.columns

    def test_saves_figure_to_file(self, tmp_path):
        out = tmp_path / "anomaly_plot.png"
        run_pipeline(output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
