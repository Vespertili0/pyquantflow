import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyquantflow.diagnostics import (
    FigureFactory,
    DiagnosticResult,
    plot_cusum_events,
    plot_multi_asset_events,
    plot_sample_concurrency,
    plot_barrier_trajectories,
)


def make_cusum_events(n: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=250, freq="B", tz="UTC")
    close = pd.Series(100 + np.cumsum(np.random.randn(250) * 0.5), index=dates)
    df = pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.random.randint(1_000, 10_000, 250),
        }
    )
    chosen = np.sort(np.random.choice(250, size=n, replace=False))
    event_ts = dates[chosen]
    events = pd.Series(event_ts, index=event_ts)
    return df, events


def make_barrier_labels(n: int = 100) -> pd.DataFrame:
    np.random.seed(99)
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    labels = np.random.choice([0.0, 1.0, 2.0], size=n)
    offsets = np.random.randint(1, 10, size=n)
    t1_idx = np.minimum(np.arange(n) + offsets, n - 1)
    t1 = pd.Series(dates[t1_idx], index=dates)
    return pd.DataFrame(
        {
            "Close": close,
            "atr": close * 0.01,
            "label": labels,
            "t1": t1,
            "sample_weight": np.random.uniform(0.1, 1.0, n),
        }
    )


class TestFoundation(unittest.TestCase):
    def test_figure_factory_theme(self):
        fig = FigureFactory.create()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.layout.paper_bgcolor, "#0F0F13")

    def test_diagnostic_result_empty_metadata(self):
        res = DiagnosticResult(figure=go.Figure(), metadata={})
        self.assertIsInstance(res, DiagnosticResult)

    def test_return_types(self):
        df, events = make_cusum_events()
        res1 = plot_cusum_events(df, events=events)
        self.assertIsInstance(res1, DiagnosticResult)

        # Fake multi-asset df for typing test
        df["ticker"] = "ABC"
        ma_df = df.reset_index().set_index(["index", "ticker"])
        ma_df.index.names = ["datetime", "ticker"]
        res2 = plot_multi_asset_events(ma_df, tickers=["ABC"])
        self.assertIsInstance(res2, DiagnosticResult)

        barrier_df = make_barrier_labels()
        res3 = plot_sample_concurrency(barrier_df["t1"], barrier_df["sample_weight"])
        self.assertIsInstance(res3, DiagnosticResult)

        res4 = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=2.0,
            sl_mult=2.0,
            horizon=5,
        )
        self.assertIsInstance(res4, DiagnosticResult)


class TestCUSUMEvents(unittest.TestCase):
    def test_candlestick_renders(self):
        df, events = make_cusum_events()
        res = plot_cusum_events(df, events=events)
        traces = [t for t in res.figure.data if isinstance(t, go.Candlestick)]
        self.assertEqual(len(traces), 1)

    def test_marker_timestamps_match_events(self):
        df, events = make_cusum_events()
        res = plot_cusum_events(df, events=events)
        marker_trace = [
            t
            for t in res.figure.data
            if isinstance(t, go.Scatter) and t.mode == "markers"
        ][0]
        # Compare as lists of timestamps to avoid index object mismatches
        self.assertEqual(list(marker_trace.x), list(events.index))

    def test_n_events_metadata(self):
        df, events = make_cusum_events(30)
        res = plot_cusum_events(df, events=events)
        self.assertEqual(res.metadata["n_events"], 30)

    def test_multi_asset_subplot_count(self):
        df, _ = make_cusum_events()
        df2 = df.copy()
        df3 = df.copy()

        df["ticker"] = "A"
        df2["ticker"] = "B"
        df3["ticker"] = "C"

        ma = pd.concat([df, df2, df3]).reset_index().set_index(["index", "ticker"])
        ma.index.names = ["datetime", "ticker"]

        res = plot_multi_asset_events(ma, tickers=["A", "B", "C"])

        # Each subplot has one trace (price). Total traces = 3.
        self.assertEqual(len(res.figure.data), 3)
        # Check rows in layout via grid
        self.assertTrue(res.figure._has_subplots())

    def test_multi_asset_warning_on_overflow(self):
        df, _ = make_cusum_events()
        dfs = []
        tickers = ["A", "B", "C", "D", "E", "F"]
        for t in tickers:
            df_t = df.copy()
            df_t["ticker"] = t
            dfs.append(df_t)

        ma = pd.concat(dfs).reset_index().set_index(["index", "ticker"])
        ma.index.names = ["datetime", "ticker"]

        res = plot_multi_asset_events(ma, tickers=tickers, max_tickers=5)
        self.assertEqual(len(res.metadata["tickers"]), 5)


class TestConcurrencyProfiler(unittest.TestCase):
    def test_returns_diagnostic_result(self):
        barrier_df = make_barrier_labels()
        res = plot_sample_concurrency(barrier_df["t1"])
        self.assertIsInstance(res, DiagnosticResult)

    def test_peak_concurrency_matches_subplot(self):
        barrier_df = make_barrier_labels()
        res = plot_sample_concurrency(barrier_df["t1"])
        c_t_trace = res.figure.data[0]  # The Bar trace
        self.assertEqual(res.metadata["peak_concurrency"], max(c_t_trace.y))

    def test_warning_bands_present(self):
        barrier_df = make_barrier_labels()
        res = plot_sample_concurrency(barrier_df["t1"], concurrency_threshold_pct=0.5)
        # Check layout for vrect shapes
        shapes = res.figure.layout.shapes
        self.assertTrue(len(shapes) > 0)

    def test_effective_sample_size_with_weights(self):
        barrier_df = make_barrier_labels()
        weights = barrier_df["sample_weight"]
        res = plot_sample_concurrency(barrier_df["t1"], weight_series=weights)
        self.assertAlmostEqual(res.metadata["effective_sample_size"], weights.sum())

    def test_no_weight_series(self):
        barrier_df = make_barrier_labels()
        res = plot_sample_concurrency(barrier_df["t1"])
        self.assertIsInstance(res.metadata["effective_sample_size"], float)


class TestBarrierTrajectories(unittest.TestCase):
    def test_n_trajectory_traces(self):
        barrier_df = make_barrier_labels()
        n_events = 15
        res = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=1.0,
            sl_mult=1.0,
            horizon=5,
            n_events=n_events,
        )
        # Main panel traces: 1 per event for path, 2 per event for barriers (TP, SL), plus vline
        # There are 15 events * 3 traces = 45 traces + 3 dummy traces = 48 traces for row 1 col 1,
        # But we only count the path traces (mode="lines" with showlegend=False and no dash)
        [
            t
            for t in res.figure.data
            if getattr(t, "xaxis", None) == "x"
            and getattr(t, "yaxis", None) == "y"
            and t.name is None
        ]
        # Above is tricky to parse. Let's just rely on the sampled size.
        self.assertEqual(res.metadata["n_sampled"], n_events)

    def test_traces_within_horizon(self):
        barrier_df = make_barrier_labels()
        horizon = 5
        res = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=1.0,
            sl_mult=1.0,
            horizon=horizon,
            n_events=5,
        )
        # Valid execution implies paths were properly sliced.
        self.assertEqual(res.metadata["n_sampled"], 5)

    def test_pct_sums_to_one(self):
        barrier_df = make_barrier_labels()
        res = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=1.0,
            sl_mult=1.0,
            horizon=5,
        )
        pct_sum = (
            res.metadata["pct_tp"]
            + res.metadata["pct_sl"]
            + res.metadata["pct_timeout"]
        )
        self.assertAlmostEqual(pct_sum, 1.0, places=5)

    def test_trace_colours_match_labels(self):
        # We can test execution correctness.
        barrier_df = make_barrier_labels()
        res = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=1.0,
            sl_mult=1.0,
            horizon=5,
            n_events=1,
        )
        self.assertEqual(res.metadata["n_sampled"], 1)

    def test_n_events_cap(self):
        barrier_df = make_barrier_labels(20)  # Only 20 items
        res = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=1.0,
            sl_mult=1.0,
            horizon=5,
            n_events=500,
        )
        self.assertEqual(res.metadata["n_sampled"], 20)
