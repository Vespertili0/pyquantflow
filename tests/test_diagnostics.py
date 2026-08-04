import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyquantflow.diagnostics import (
    FigureFactory,
    DiagnosticResult,
    plot_cusum_events,
    plot_multi_asset_events,
    plot_sample_concurrency,
    plot_barrier_trajectories,
    plot_downsampling_shift,
    plot_stationarity_profile,
    plot_feature_clusters,
    plot_cv_splits,
    plot_fold_feature_drift,
    plot_meta_label_entropy,
    plot_meta_label_precision_recall,
    plot_sadf_regimes,
    plot_cpcv_paths,
)

from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.data.features.fractional_differentiation import frac_diff_ffd
from pyquantflow.model.feature_evaluation import StationaryTransformer, FeatureEvaluator
from pyquantflow.model.cross_validation import PurgedKFoldCV, CombinatorialPurgedKFold
from pyquantflow.model.classifier import PrimarySecondaryClassifier
from pyquantflow.data.sk_transformers import GSADFTransformer


# ==============================================================================
# Synthetic Data Fixtures
# ==============================================================================


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


def make_feature_matrix(n_samples=500, n_features=8):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")

    data = {}
    for i in range(n_features):
        data[f"feat_{i}"] = np.cumsum(np.random.randn(n_samples))

    raw_df = pd.DataFrame(data, index=dates)

    # Event DF is a 40% random subset with a distribution shift
    event_idx = np.sort(
        np.random.choice(n_samples, size=int(n_samples * 0.4), replace=False)
    )
    event_df = raw_df.iloc[event_idx].copy()

    # Add shift
    for col in event_df.columns:
        event_df[col] = event_df[col] + np.random.randn(len(event_df)) * 2

    return raw_df, event_df


def make_ffd_pair(n=300, d=0.4):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    raw_series = pd.Series(np.cumsum(np.random.randn(n)), index=dates, name="price")
    ffd_series = frac_diff_ffd(raw_series, d=d)
    return raw_series, ffd_series


def make_importance_results(n_clusters=4, n_features=8):
    # Returns Dict[int, Dict[str, pd.DataFrame]]
    sfi_rows = []
    mda_rows = []

    features_per_cluster = n_features // n_clusters
    for c_id in range(1, n_clusters + 1):
        feats = [f"f_{c_id}_{i}" for i in range(features_per_cluster)]
        sfi_rows.append(
            {
                "cluster_id": c_id,
                "features": ", ".join(feats),
                "sfi_mean": 0.05 * c_id,
                "sfi_std": 0.01,
            }
        )
        mda_rows.append(
            {
                "cluster_id": c_id,
                "features": ", ".join(feats),
                "mda_mean": 0.1 * c_id,
                "mda_std": 0.02,
            }
        )

    sfi_df = pd.DataFrame(sfi_rows).set_index("cluster_id")
    mda_df = pd.DataFrame(mda_rows).set_index("cluster_id")

    return {0: {"SFI": sfi_df, "MDA": mda_df}}


def make_cv_fixtures(n_samples=300, n_splits=5):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    X = pd.DataFrame({"f1": np.random.randn(n_samples)}, index=dates)
    y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)

    # Clean t1: each event ends 2 bars later
    t1_clean = pd.Series(
        dates[np.minimum(np.arange(n_samples) + 2, n_samples - 1)], index=dates
    )

    # Leaking t1: some events end 50 bars later (causing overlap into test set)
    t1_leaking = pd.Series(
        dates[np.minimum(np.arange(n_samples) + 50, n_samples - 1)], index=dates
    )

    cv_clean = PurgedKFoldCV(n_splits=n_splits, t1=t1_clean, embargo_pct=0.01)

    # Dummy splitter that fails to purge for auditing test
    cv_leaking = PurgedKFoldCV(n_splits=n_splits, t1=t1_leaking, embargo_pct=0.01)

    def leaking_split(X, y=None):
        fold_size = n_samples // n_splits
        for k in range(n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < n_splits - 1 else n_samples
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate(
                [np.arange(0, test_start), np.arange(test_end, n_samples)]
            )
            yield train_idx, test_idx

    cv_leaking.split = leaking_split

    return cv_clean, cv_leaking, X, y


def make_meta_predictions(n_events=200):
    np.random.seed(42)
    return pd.DataFrame(
        {
            "primary_pred": np.random.randint(0, 2, n_events),
            "primary_proba": np.random.uniform(0.4, 0.6, n_events),
            "primary_entropy": np.random.uniform(0.0, 1.0, n_events),
            "secondary_proba": np.random.uniform(0.0, 1.0, n_events),
            "final_decision": np.random.randint(0, 2, n_events),
            "label": np.random.choice([0, 1, 2], n_events),
        }
    )


def make_sadf_series(n_bars=300, n_bubbles=3):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    price_series = pd.Series(np.cumsum(np.random.randn(n_bars)), index=dates)

    sadf_vals = np.random.uniform(0.5, 1.2, n_bars)
    bubble_centers = np.linspace(50, n_bars - 50, n_bubbles, dtype=int)
    for c in bubble_centers:
        sadf_vals[c - 5 : c + 5] = np.random.uniform(1.5, 2.5, 10)

    sadf_series = pd.Series(sadf_vals, index=dates)
    return price_series, sadf_series


def make_population(n_paths=20, path_length=252):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=path_length, freq="B")

    population = []
    for _ in range(n_paths):
        mock_portfolio = MagicMock()
        mock_portfolio.returns = pd.Series(
            np.random.normal(0.0005, 0.01, path_length), index=dates
        )
        mock_portfolio.annualized_sharpe_ratio = np.random.normal(1.0, 0.5)
        population.append(mock_portfolio)

    return population


# ==============================================================================
# Unit Tests
# ==============================================================================


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
        self.assertEqual(len(res.figure.data), 3)
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
        c_t_trace = res.figure.data[0]
        self.assertEqual(res.metadata["peak_concurrency"], max(c_t_trace.y))

    def test_warning_bands_present(self):
        barrier_df = make_barrier_labels()
        res = plot_sample_concurrency(barrier_df["t1"], concurrency_threshold_pct=0.5)
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
        barrier_df = make_barrier_labels(20)
        res = plot_barrier_trajectories(
            barrier_df,
            event_timestamps=barrier_df.index,
            pt_mult=1.0,
            sl_mult=1.0,
            horizon=5,
            n_events=500,
        )
        self.assertEqual(res.metadata["n_sampled"], 20)


class TestFeaturesDownsampling(unittest.TestCase):
    def test_downsampling_shift(self):
        raw_df, event_df = make_feature_matrix(n_features=3)
        cols = ["feat_0", "feat_1", "feat_2"]
        res = plot_downsampling_shift(raw_df, event_df, cols)

        self.assertIsInstance(res, DiagnosticResult)
        self.assertEqual(len(res.figure.data), 6)

        self.assertIn("divergence_scores", res.metadata)
        self.assertIn("n_features_flagged", res.metadata)
        self.assertEqual(len(res.metadata["divergence_scores"]), 3)
        self.assertIsInstance(res.metadata["divergence_scores"]["feat_0"], float)


class TestFeaturesStationarity(unittest.TestCase):
    def test_stationarity_profile(self):
        raw, ffd = make_ffd_pair()
        max_lags = 20
        res = plot_stationarity_profile(
            raw, ffd, d_order=0.4, ticker="TEST", max_lags=max_lags
        )

        self.assertIsInstance(res, DiagnosticResult)
        self.assertEqual(len(res.figure.data), 4)

        self.assertIn("adf_stat", res.metadata)
        self.assertIn("d_order", res.metadata)
        self.assertIn("lag1_acf_raw", res.metadata)
        self.assertIn("lag1_acf_ffd", res.metadata)

        bar_traces = [t for t in res.figure.data if isinstance(t, go.Bar)]
        self.assertEqual(len(bar_traces[0].x), max_lags + 1)


class TestClustering(unittest.TestCase):
    def test_feature_clusters(self):
        regime_results = make_importance_results(n_clusters=2, n_features=4)
        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )

        res = plot_feature_clusters(regime_results, corr_matrix, regime_id=0)
        self.assertIsInstance(res, DiagnosticResult)

        self.assertEqual(len(res.figure.data), 5)
        self.assertIn("cluster_assignments", res.metadata)
        self.assertIn("top_cluster_id", res.metadata)
        self.assertIn("top_cluster_mda", res.metadata)
        self.assertEqual(res.metadata["n_clusters"], 2)
        self.assertEqual(res.metadata["n_features"], 4)


class TestCVSplits(unittest.TestCase):
    def test_cv_splits_clean(self):
        cv_clean, _, X, y = make_cv_fixtures(n_splits=3)
        res = plot_cv_splits(cv_clean, X, y)
        self.assertIsInstance(res, DiagnosticResult)
        self.assertFalse(res.metadata["has_leakage"])
        self.assertEqual(res.metadata["n_splits"], 3)

    @patch("pyquantflow.diagnostics.cv.warnings.warn")
    def test_cv_splits_leaking(self, mock_warn):
        _, cv_leaking, X, y = make_cv_fixtures(n_splits=3)
        res = plot_cv_splits(cv_leaking, X, y)

        self.assertTrue(res.metadata["has_leakage"])
        self.assertTrue(len(res.metadata["leaking_fold_indices"]) > 0)
        self.assertTrue(mock_warn.called)


class TestCVFeatureDrift(unittest.TestCase):
    def test_feature_drift(self):
        cv_clean, _, X, y = make_cv_fixtures(n_splits=4)
        res = plot_fold_feature_drift(X, cv_clean, "f1")

        self.assertIsInstance(res, DiagnosticResult)
        self.assertEqual(res.metadata["n_splits"], 4)
        self.assertEqual(res.metadata["feature_col"], "f1")
        self.assertEqual(len(res.figure.data), 4)


class TestMetaLabelEntropy(unittest.TestCase):
    def test_meta_label_entropy(self):
        df = make_meta_predictions()
        res = plot_meta_label_entropy(df)

        self.assertIsInstance(res, DiagnosticResult)
        self.assertIn("entropy_return_spearman", res.metadata)
        self.assertIn("meta_filter_rate", res.metadata)
        self.assertIn("median_entropy_passed", res.metadata)
        self.assertEqual(len(res.figure.data), 5)


class TestMetaLabelPrecisionRecall(unittest.TestCase):
    def test_meta_label_pr(self):
        df = make_meta_predictions()
        y_true = (df["label"] == 2).astype(int)

        res = plot_meta_label_precision_recall(
            y_true, df["primary_proba"], df["final_decision"], df["secondary_proba"]
        )

        self.assertIsInstance(res, DiagnosticResult)
        self.assertIn("primary_auc_pr", res.metadata)
        self.assertIn("meta_auc_pr", res.metadata)
        self.assertEqual(len(res.figure.data), 2)


class TestSADFRegimes(unittest.TestCase):
    def test_sadf_regimes(self):
        price, sadf = make_sadf_series()
        events = pd.DatetimeIndex(np.random.choice(price.index, 20, replace=False))

        res = plot_sadf_regimes(price, sadf, critical_value=1.4, events=events)

        self.assertIsInstance(res, DiagnosticResult)
        self.assertIn("pct_explosive_regime", res.metadata)
        self.assertIn("n_events_in_explosive", res.metadata)
        self.assertEqual(len(res.figure.data), 3)
        self.assertTrue(len(res.figure.layout.shapes) > 1)


class TestPBOPaths(unittest.TestCase):
    def test_pbo_paths(self):
        population = make_population(n_paths=10)
        res = plot_cpcv_paths(population, benchmark_sharpe=0.5)

        self.assertIsInstance(res, DiagnosticResult)
        self.assertEqual(res.metadata["n_paths"], 10)
        self.assertIn("pbo_estimate", res.metadata)
        self.assertEqual(len(res.figure.data), 14)


class TestAccessors(unittest.TestCase):
    def test_asset_organiser_accessors_bound(self):
        self.assertTrue(hasattr(AssetOrganiser, "plot_cusum_events"))
        self.assertTrue(hasattr(AssetOrganiser, "plot_sample_concurrency"))

    def test_phase2_accessors_bound(self):
        self.assertTrue(hasattr(StationaryTransformer, "plot_stationarity_profile"))
        self.assertTrue(hasattr(FeatureEvaluator, "plot_feature_clusters"))
        self.assertTrue(hasattr(PurgedKFoldCV, "plot_splits"))
        self.assertTrue(hasattr(CombinatorialPurgedKFold, "plot_splits"))
        self.assertTrue(hasattr(PrimarySecondaryClassifier, "plot_meta_diagnostics"))
        self.assertTrue(hasattr(GSADFTransformer, "plot_sadf_regimes"))


class TestClusteringCoverage(unittest.TestCase):
    def test_regime_dict_no_regime_id(self):
        regime_results = make_importance_results(n_clusters=2, n_features=4)
        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )
        res = plot_feature_clusters(regime_results, corr_matrix, regime_id=None)
        self.assertIsNotNone(res)

    def test_regime_dict_bad_regime_id(self):
        regime_results = make_importance_results(n_clusters=2, n_features=4)
        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )
        with self.assertRaises(KeyError):
            plot_feature_clusters(regime_results, corr_matrix, regime_id=999)

    def test_dataframe_with_regime_id(self):
        regime_results = make_importance_results(n_clusters=2, n_features=4)
        dfs = []
        for r_id, r_data in regime_results.items():
            df = r_data["SFI"].join(r_data["MDA"][["mda_mean", "mda_std"]], how="outer")
            df["regime"] = r_id
            dfs.append(df)
        df_all = pd.concat(dfs).reset_index().set_index(["regime", "cluster_id"])

        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )
        res = plot_feature_clusters(df_all, corr_matrix, regime_id=0)
        self.assertIsNotNone(res)

    def test_dataframe_no_regime_id(self):
        regime_results = make_importance_results(n_clusters=2, n_features=4)
        dfs = []
        for r_id, r_data in regime_results.items():
            df = r_data["SFI"].join(r_data["MDA"][["mda_mean", "mda_std"]], how="outer")
            df["regime"] = r_id
            dfs.append(df)
        df_all = pd.concat(dfs).reset_index().set_index(["regime", "cluster_id"])
        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )
        res = plot_feature_clusters(df_all, corr_matrix, regime_id=None)
        self.assertIsNotNone(res)

    def test_empty_dataframe(self):
        df_all = pd.DataFrame(
            columns=[
                "regime",
                "cluster_id",
                "features",
                "sfi_mean",
                "sfi_std",
                "mda_mean",
                "mda_std",
            ]
        ).set_index(["regime", "cluster_id"])
        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )
        res = plot_feature_clusters(df_all, corr_matrix, regime_id=None)
        self.assertEqual(res.metadata["top_cluster_id"], -1)

    def test_provided_linkage_matrix(self):
        regime_results = make_importance_results(n_clusters=2, n_features=4)
        corr_matrix = pd.DataFrame(
            np.eye(4),
            columns=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
            index=["f_1_0", "f_1_1", "f_2_0", "f_2_1"],
        )
        dist = np.sqrt(0.5 * (1 - corr_matrix.clip(-1, 1)))
        import scipy.spatial.distance as ssd
        import scipy.cluster.hierarchy as sch

        condensed = ssd.squareform(dist.values, checks=False)
        linkage_matrix = sch.linkage(condensed, method="ward")
        res = plot_feature_clusters(
            regime_results, corr_matrix, linkage_matrix=linkage_matrix, regime_id=0
        )
        self.assertIsNotNone(res)


class TestCVCoverage(unittest.TestCase):
    def test_cpcv_splits(self):
        cpcv = CombinatorialPurgedKFold(n_splits=4, n_test_splits=2)
        X = pd.DataFrame(
            np.random.randn(100, 2), index=pd.date_range("2020-01-01", periods=100)
        )
        y = pd.Series(np.random.randint(0, 2, 100), index=X.index)
        res = plot_cv_splits(cpcv, X, y)
        self.assertIsNotNone(res)

    def test_pkf_t1_as_string(self):
        X = pd.DataFrame(
            np.random.randn(100, 2), index=pd.date_range("2020-01-01", periods=100)
        )
        X["t1_col"] = X.index + pd.Timedelta(days=2)
        pkf = PurgedKFoldCV(n_splits=3, t1="t1_col", embargo_pct=0.01)
        y = pd.Series(np.random.randint(0, 2, 100), index=X.index)
        res = plot_cv_splits(pkf, X, y)
        self.assertIsNotNone(res)

    def test_pkf_t1_none(self):
        X = pd.DataFrame(
            np.random.randn(100, 2), index=pd.date_range("2020-01-01", periods=100)
        )
        pkf = PurgedKFoldCV(n_splits=3, t1=None)
        y = pd.Series(np.random.randint(0, 2, 100), index=X.index)
        res = plot_cv_splits(pkf, X, y)
        self.assertIsNotNone(res)

    def test_missing_train_test(self):
        cv = PurgedKFoldCV(n_splits=2)

        def bad_split(X, y=None):
            yield [], []

        cv.split = bad_split
        X = pd.DataFrame(
            np.random.randn(10, 2), index=pd.date_range("2020-01-01", periods=10)
        )
        y = pd.Series(np.random.randint(0, 2, 10), index=X.index)
        res = plot_cv_splits(cv, X, y)
        self.assertIsNotNone(res)


class TestFeaturesCoverage(unittest.TestCase):
    def test_wasserstein_metric(self):
        raw_df, event_df = make_feature_matrix(n_features=2)
        res = plot_downsampling_shift(
            raw_df, event_df, ["feat_0"], divergence_metric="wasserstein"
        )
        self.assertEqual(res.metadata["divergence_metric"], "wasserstein")

    def test_bad_divergence_metric(self):
        raw_df, event_df = make_feature_matrix(n_features=2)
        with self.assertRaises(ValueError):
            plot_downsampling_shift(
                raw_df, event_df, ["feat_0"], divergence_metric="invalid"
            )

    def test_short_data(self):
        raw_df = pd.DataFrame({"f1": [1.0]})
        event_df = pd.DataFrame({"f1": [1.0]})
        res = plot_downsampling_shift(raw_df, event_df, ["f1"])
        self.assertTrue(np.isnan(res.metadata["divergence_scores"]["f1"]))


class TestRegimesCoverage(unittest.TestCase):
    def test_events_as_series(self):
        price, sadf = make_sadf_series()
        events_idx = pd.DatetimeIndex(np.random.choice(price.index, 5, replace=False))
        events_series = pd.Series(events_idx, index=events_idx)
        res = plot_sadf_regimes(price, sadf, events=events_series)
        self.assertIsNotNone(res)

    def test_events_none(self):
        price, sadf = make_sadf_series()
        res = plot_sadf_regimes(price, sadf, events=None, title="Custom Title")
        self.assertIsNotNone(res)
        self.assertEqual(res.figure.layout.title.text, "Custom Title")

    def test_empty_events(self):
        price, sadf = make_sadf_series()
        res = plot_sadf_regimes(price, sadf, events=pd.DatetimeIndex([]))
        self.assertIsNotNone(res)

    def test_empty_sadf(self):
        price, sadf = make_sadf_series(n_bars=0, n_bubbles=0)
        res = plot_sadf_regimes(price, sadf)
        self.assertTrue(np.isnan(res.metadata["max_sadf_stat"]))


class TestAccessorsCoverage(unittest.TestCase):
    def test_ao_cusum_events(self):
        df, events = make_cusum_events()
        multi_asset = df.reset_index().set_index(["index"])
        multi_asset["ticker"] = "ABC"
        multi_asset = multi_asset.reset_index().set_index(["index", "ticker"])
        multi_asset.index.names = ["datetime", "ticker"]
        ao = AssetOrganiser(
            multi_asset=multi_asset, cutoff_date="2020-01-01", target_features=["Close"]
        )
        ao.cusum_events_map = {"ABC": events}
        res = ao.plot_cusum_events()
        self.assertIsNotNone(res)

    def test_ao_cusum_events_error(self):
        df, _ = make_cusum_events()
        multi_asset = df.reset_index().set_index(["index"])
        multi_asset["ticker"] = "ABC"
        multi_asset = multi_asset.reset_index().set_index(["index", "ticker"])
        multi_asset.index.names = ["datetime", "ticker"]
        ao = AssetOrganiser(
            multi_asset=multi_asset, cutoff_date="2020-01-01", target_features=["Close"]
        )
        with self.assertRaises(AttributeError):
            ao.plot_cusum_events()

    def test_ao_sample_concurrency(self):
        barrier_df = make_barrier_labels()
        barrier_df = barrier_df.reset_index().rename(columns={"index": "datetime"})
        barrier_df["ticker"] = "ABC"
        barrier_df = barrier_df.set_index(["datetime", "ticker"])
        ao = AssetOrganiser(
            multi_asset=barrier_df, cutoff_date="2020-01-01", target_features=["Close"]
        )
        ao.weight_col = "sample_weight"
        res = ao.plot_sample_concurrency()
        self.assertIsNotNone(res)

    def test_ao_sample_concurrency_error(self):
        df = pd.DataFrame(
            {"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )
        df = df.reset_index().rename(columns={"index": "datetime"})
        df["ticker"] = "ABC"
        df = df.set_index(["datetime", "ticker"])
        ao = AssetOrganiser(
            multi_asset=df, cutoff_date="2020-01-01", target_features=["Close"]
        )
        with self.assertRaises(KeyError):
            ao.plot_sample_concurrency()

    @patch("pyquantflow.diagnostics.features.plot_stationarity_profile")
    @patch("pyquantflow.data.features.fractional_differentiation.adf_screened_ffd")
    def test_st_stationarity_profile(self, mock_ffd, mock_plot):
        mock_ffd.return_value = (pd.Series([1, 2]), pd.Series([1, 2]))
        st = StationaryTransformer()
        st.optimal_d_ = {"ABC": 0.5}
        st.ffd_thres = 1e-5
        st.plot_stationarity_profile(pd.Series([1, 2, 3]), "ABC")
        mock_plot.assert_called_once()

    @patch("pyquantflow.diagnostics.clustering.plot_feature_clusters")
    def test_fe_feature_clusters(self, mock_plot):
        fe = FeatureEvaluator(features=["f1"])
        fe.raw_features = ["f2"]
        fe.importance_df = pd.DataFrame()
        df = pd.DataFrame({"f1": [1, 2], "f2": [2, 3]})
        fe.plot_feature_clusters(df)
        mock_plot.assert_called_once()

    @patch("pyquantflow.diagnostics.clustering.plot_feature_clusters")
    def test_fe_feature_clusters_no_importance(self, mock_plot):
        fe = FeatureEvaluator(features=["f1"])
        fe.raw_features = ["f2"]
        fe.importance_df = None
        df = pd.DataFrame({"f1": [1, 2], "f2": [2, 3]})
        
        with self.assertRaisesRegex(ValueError, "importance_df is None"):
            fe.plot_feature_clusters(df)

    @patch("pyquantflow.diagnostics.cv.plot_cv_splits")
    def test_cv_splits_accessors(self, mock_plot):
        pkf = PurgedKFoldCV(n_splits=2)
        cpcv = CombinatorialPurgedKFold(n_splits=3, n_test_splits=1)
        X = pd.DataFrame()
        y = pd.Series()
        pkf.plot_splits(X, y)
        cpcv.plot_splits(X, y)
        self.assertEqual(mock_plot.call_count, 2)

    @patch("pyquantflow.diagnostics.metalabel.plot_meta_label_entropy")
    def test_psc_meta_diagnostics(self, mock_plot):
        psc = PrimarySecondaryClassifier(
            MagicMock(), MagicMock(), primary_features=["a"], secondary_features=["a"]
        )
        psc.transform = MagicMock(return_value=pd.DataFrame({"a": [1]}))
        psc.plot_meta_diagnostics(pd.DataFrame({"a": [1]}), pd.Series([1]))
        mock_plot.assert_called_once()

    @patch("pyquantflow.diagnostics.regimes.plot_sadf_regimes")
    def test_gsadf_sadf_regimes(self, mock_plot):
        gt = GSADFTransformer()
        gt.plot_sadf_regimes(pd.Series([1]), pd.Series([1]))
        mock_plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
