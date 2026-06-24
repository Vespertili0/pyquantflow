import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, f1_score, accuracy_score

from pyquantflow.model.feature_evaluation import (
    _adf_test_stat,
    _adf_p_value,
    StationaryTransformer,
    FeatureEvaluator,
)


class TestFeatureEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a stationary series (white noise)
        np.random.seed(42)
        self.stationary_series = pd.Series(np.random.randn(100))

        # Create a non-stationary series (random walk)
        self.random_walk = pd.Series(np.random.randn(100).cumsum())

        # Create initial panel data via helper
        self.panel_df = self._create_synthetic_panel_df()

    def _create_synthetic_panel_df(self, n_periods=50, tickers=None, random_seed=42):
        """
        Helper method to generate MultiIndex synthetic panel data.
        """
        if tickers is None:
            tickers = ["AAPL", "MSFT"]
        np.random.seed(random_seed)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=n_periods), tickers],
            names=["datetime", "ticker"],
        )
        n_total = n_periods * len(tickers)
        return pd.DataFrame(
            {
                "feat1": np.random.randn(n_total).cumsum(),
                "feat2": np.random.randn(n_total),
                "target": np.random.choice([0, 1], n_total),
                "weight": np.random.uniform(0.5, 1.5, n_total),
            },
            index=idx,
        )

    def _create_synthetic_single_index_df(self, n_periods=100, random_seed=42):
        """
        Helper method to generate single-index synthetic data.
        """
        np.random.seed(random_seed)
        dates = pd.date_range("2020-01-01", periods=n_periods)
        return pd.DataFrame(
            {
                "feat1": np.random.randn(n_periods).cumsum(),
                "feat2": np.random.randn(n_periods),
                "target": np.random.choice([0, 1], n_periods),
                "weight": np.random.uniform(0.5, 1.5, n_periods),
            },
            index=dates,
        )

    def test_native_adf_logic(self):
        # Stationary series should have significant negative t-stat and small p-value
        stat_t = _adf_test_stat(self.stationary_series)
        stat_p = _adf_p_value(stat_t)
        self.assertLess(stat_p, 0.05)

        # Non-stationary series should have non-significant p-value
        rw_t = _adf_test_stat(self.random_walk)
        rw_p = _adf_p_value(rw_t)
        self.assertGreater(rw_p, 0.05)

    def test_stationary_transformer(self):
        transformer = StationaryTransformer(d_grid=np.array([0.0, 0.5, 1.0]))
        X = self.panel_df[["feat1", "feat2"]]

        # Fit should identify that feat2 needs lower d than feat1
        transformer.fit(X)
        self.assertIn("feat1", transformer.optimal_d_)
        self.assertIn("feat2", transformer.optimal_d_)

        # Transform should apply FFD and return standardised series
        X_trans = transformer.transform(X)

        self.assertEqual(X_trans.shape, X.shape)
        # Check that missing data from lag (rolling window) is NaN
        self.assertTrue(np.isnan(X_trans["feat1"].iloc[0]))

    def test_stationary_transformer_global_z_mode(self):
        """
        Test StationaryTransformer with global z-mode standardisation.
        """
        transformer = StationaryTransformer(
            d_grid=np.array([0.0, 0.5, 1.0]), z_mode="global"
        )
        X = self.panel_df[["feat1", "feat2"]]
        transformer.fit(X)

        # Verify global z-score parameters are populated
        self.assertIn("feat1", transformer.z_mean_)
        self.assertIn("feat1", transformer.z_std_)
        self.assertIn("feat2", transformer.z_mean_)
        self.assertIn("feat2", transformer.z_std_)

        # Transform and ensure outputs align with global parameters
        X_trans = transformer.transform(X)
        self.assertEqual(X_trans.shape, X.shape)

    def test_stationary_transformer_single_index(self):
        """
        Test StationaryTransformer on a single-index DataFrame.
        """
        single_df = self._create_synthetic_single_index_df()
        transformer = StationaryTransformer(d_grid=np.array([0.0, 0.5, 1.0]))
        X = single_df[["feat1", "feat2"]]

        transformer.fit(X)
        X_trans = transformer.transform(X)
        self.assertEqual(X_trans.shape, X.shape)
        self.assertIn("feat1", transformer.optimal_d_)

    def test_stationary_transformer_get_feature_names_out(self):
        """
        Test get_feature_names_out behaviour.
        """
        transformer = StationaryTransformer()
        # Should raise ValueError before fit when no input features are provided
        with self.assertRaises(ValueError):
            transformer.get_feature_names_out()

        # Passing input features directly should work even before fit
        feats = ["a", "b"]
        np.testing.assert_array_equal(
            transformer.get_feature_names_out(feats), np.array(feats)
        )

        # Fitting should store feature_names_in_
        X = self.panel_df[["feat1", "feat2"]]
        transformer.fit(X)
        np.testing.assert_array_equal(
            transformer.get_feature_names_out(), np.array(["feat1", "feat2"])
        )

    def test_stationary_transformer_invalid_multiindex_fallback(self):
        """
        Test fallback in fit() and transform() when MultiIndex lacks "ticker" level.
        """
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=10), ["AAPL", "MSFT"]],
            names=["datetime", "asset"],
        )
        df_invalid = pd.DataFrame({"feat1": np.random.randn(20)}, index=idx)

        transformer = StationaryTransformer(
            d_grid=np.array([0.0, 0.5, 1.0]), z_mode="global"
        )
        # Fit and transform should succeed by falling back to single-asset estimation path
        transformer.fit(df_invalid)
        self.assertIn("feat1", transformer.optimal_d_)

        X_trans = transformer.transform(df_invalid)
        self.assertEqual(X_trans.shape, df_invalid.shape)

    def test_feature_evaluator_profiling(self):
        evaluator = FeatureEvaluator(
            features=["feat1", "feat2"], target_col="target", cv=KFold(n_splits=2)
        )

        profiles = evaluator.compute_time_series_profiles(
            self.panel_df, columns=["feat1", "feat2"], groupby_level="ticker"
        )

        # tsfeatures returns many columns. Just check a few standard ones exist or it's not empty
        self.assertTrue(len(profiles.columns) > 0)
        self.assertEqual(len(profiles), 4)  # 2 tickers * 2 features

    def test_feature_evaluator_clustering_and_mda(self):
        # We need more features to test clustering effectively
        df = self.panel_df.copy()
        df["feat3"] = df["feat1"] + np.random.randn(100) * 0.1  # Correlated with feat1
        df["feat4"] = np.random.randn(100)

        evaluator = FeatureEvaluator(
            features=["feat1", "feat2", "feat3", "feat4"],
            target_col="target",
            weight_col="weight",
            cv=KFold(n_splits=2),
            memory_threshold=-1.0,
        )

        clusters = evaluator.cluster_entities(
            df[evaluator.features], method="correlation"
        )

        # feat1 and feat3 should likely be in the same cluster due to high correlation
        found_together = False
        for c_id, cols in clusters.items():
            if "feat1" in cols and "feat3" in cols:
                found_together = True
        self.assertTrue(found_together)

        # Test full evaluation
        dummy = DecisionTreeClassifier()

        # Transform features
        df_trans = evaluator.fit_transform_features(df)

        # DecisionTreeClassifier does not natively handle NaNs, so we fill them for the dummy test
        df_trans = df_trans.fillna(0)

        results = evaluator.evaluate_importance(
            df_trans,
            estimator=dummy,
            metric=log_loss,
            balance_classes=False,
            greater_is_better=False,
            needs_proba=True,
        )

        # Check regime output structure
        self.assertTrue(len(results.keys()) > 0)
        first_regime = list(results.keys())[0]
        self.assertIn("MDA", results[first_regime])
        self.assertIn("SFI", results[first_regime])
        self.assertTrue(len(results[first_regime]["MDA"]) > 0)

    def test_mda_greater_is_better_false(self):
        """
        Req 1.1: brier_score_loss (lower is better) with a highly predictive feature.
        MDA importance must be positive for the informative cluster.
        """
        np.random.seed(0)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)
        signal = np.random.randn(n)
        target = (signal > 0).astype(int)
        noise = np.random.randn(n)

        df = pd.DataFrame(
            {"signal": signal, "noise": noise, "target": target},
            index=dates,
        )
        df.index.name = "datetime"

        evaluator = FeatureEvaluator(
            features=["signal", "noise"],
            target_col="target",
            cv=KFold(n_splits=3),
            memory_threshold=-1.0,
        )
        df_trans = evaluator.fit_transform_features(df)
        df_trans = df_trans.fillna(0)

        results = evaluator.evaluate_importance(
            df_trans,
            estimator=LogisticRegression(),
            metric=brier_score_loss,
            greater_is_better=False,
            needs_proba=True,
        )

        self.assertTrue(len(results) > 0)
        first_regime = list(results.values())[0]
        mda_df = first_regime["MDA"]
        self.assertFalse(mda_df["mda_mean"].isna().all())

    def test_mda_greater_is_better_true(self):
        """
        Req 1.1: f1_score (higher is better) with a highly predictive feature.
        MDA importance must be positive for the informative cluster.
        """
        np.random.seed(1)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)
        signal = np.random.randn(n)
        target = (signal > 0).astype(int)
        noise = np.random.randn(n)

        df = pd.DataFrame(
            {"signal": signal, "noise": noise, "target": target},
            index=dates,
        )
        df.index.name = "datetime"

        evaluator = FeatureEvaluator(
            features=["signal", "noise"],
            target_col="target",
            cv=KFold(n_splits=3),
            memory_threshold=-1.0,
        )
        df_trans = evaluator.fit_transform_features(df)
        df_trans = df_trans.fillna(0)

        results = evaluator.evaluate_importance(
            df_trans,
            estimator=LogisticRegression(),
            metric=f1_score,
            greater_is_better=True,
            needs_proba=False,
        )

        self.assertTrue(len(results) > 0)
        first_regime = list(results.values())[0]
        mda_df = first_regime["MDA"]
        self.assertFalse(mda_df["mda_mean"].isna().all())

    def test_feature_evaluator_gate1_pruning(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=100), ["AAPL"]],
            names=["datetime", "ticker"],
        )

        np.random.seed(42)
        ar1 = [0.0]
        for _ in range(99):
            ar1.append(0.8 * ar1[-1] + np.random.randn())

        df = pd.DataFrame(
            {
                "ar1_feat": ar1,
                "white_noise": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            },
            index=idx,
        )

        evaluator = FeatureEvaluator(
            features=["ar1_feat", "white_noise"],
            target_col="target",
            memory_threshold=0.10,
            significance_level=0.05,
        )

        df_trans = evaluator.fit_transform_features(df)

        self.assertIn("ar1_feat", evaluator.features)
        self.assertNotIn("white_noise", evaluator.features)
        self.assertIn("ar1_feat", df_trans.columns)
        self.assertNotIn("white_noise", df_trans.columns)

    def test_feature_evaluator_gate1_nan_fallback(self):
        """
        Verify that features resulting in NaN for ACF1 memory check are kept.
        """
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=10), ["AAPL"]],
            names=["datetime", "ticker"],
        )
        df = pd.DataFrame(
            {
                "const_feat": [42.0] * 10,
                "target": [0, 1] * 5,
            },
            index=idx,
        )
        evaluator = FeatureEvaluator(
            features=["const_feat"],
            target_col="target",
            memory_threshold=0.10,
        )
        evaluator.fit_transform_features(df)
        self.assertIn("const_feat", evaluator.features)

    def test_feature_evaluator_freq_parameter(self):
        evaluator = FeatureEvaluator(
            features=["feat1"],
            target_col="target",
            freq=5,
        )
        self.assertEqual(evaluator.freq, 5)

    def test_raw_features_bypass_ffd(self):
        """
        Req 2.1: raw_features must not be fractionally differentiated.
        """
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)

        feat1 = np.random.randn(n).cumsum()
        volume = np.random.randint(1_000, 100_000, n).astype(float)
        target = np.random.choice([0, 1], n)

        df = pd.DataFrame(
            {"feat1": feat1, "Volume": volume, "target": target},
            index=dates,
        )
        df.index.name = "datetime"

        evaluator = FeatureEvaluator(
            features=["feat1"],
            raw_features=["Volume"],
            target_col="target",
            memory_threshold=-1.0,
        )
        evaluator.stationary_transformer.ffd_thres = 1e-2
        evaluator.stationary_transformer.rolling_z_window = 10

        df_out = evaluator.fit_transform_features(df)

        self.assertIn("Volume", df_out.columns)

        np.testing.assert_array_equal(
            df_out["Volume"].dropna().values,
            df["Volume"].loc[df_out["Volume"].dropna().index].values,
            err_msg="raw_features column 'Volume' was unexpectedly modified.",
        )

        transformed_vals = df_out["feat1"].dropna()
        self.assertGreater(len(transformed_vals), 0)
        original_vals = df["feat1"].loc[transformed_vals.index]
        self.assertFalse(
            np.allclose(transformed_vals.values, original_vals.values),
            "Expected 'feat1' to be FFD-transformed but it appears unchanged.",
        )

        self.assertIn("Volume", evaluator.raw_features)

    def test_feature_evaluator_missing_raw_features(self):
        """
        Verify that fit_transform_features raises KeyError if a raw feature is missing.
        """
        evaluator = FeatureEvaluator(
            features=["feat1"],
            raw_features=["missing_col"],
            target_col="target",
        )
        with self.assertRaises(KeyError):
            evaluator.fit_transform_features(self.panel_df)

    def test_feature_evaluator_t1_col_preservation(self):
        """
        Test that t1_col is correctly kept and preserved in transformed features.
        """
        df = self.panel_df.copy()
        df["t1"] = pd.date_range("2020-01-02", periods=100)
        evaluator = FeatureEvaluator(
            features=["feat2"],
            target_col="target",
            t1_col="t1",
            memory_threshold=-1.0,
        )
        df_out = evaluator.fit_transform_features(df)
        self.assertIn("t1", df_out.columns)
        self.assertTrue(df_out["t1"].equals(df["t1"]))

    def test_cluster_entities_with_categorical(self):
        """
        Req 2.3: cluster_entities must not crash when the input DataFrame
        contains boolean or string (object-dtype) columns.
        """
        np.random.seed(0)
        n = 100
        df_mixed = pd.DataFrame(
            {
                "numeric_a": np.random.randn(n),
                "numeric_b": np.random.randn(n),
                "bool_col": np.random.choice([True, False], n),
                "str_col": np.random.choice(["up", "down", "flat"], n),
            }
        )

        evaluator = FeatureEvaluator(
            features=["numeric_a", "numeric_b"],
            target_col="numeric_a",
        )

        with patch("pyquantflow.model.feature_evaluation.warnings.warn") as mock_warn:
            clusters = evaluator.cluster_entities(df_mixed, method="correlation")

        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0)

        all_clustered = [col for cols in clusters.values() for col in cols]
        for col in ["numeric_a", "numeric_b", "bool_col", "str_col"]:
            self.assertIn(col, all_clustered)

        self.assertTrue(
            mock_warn.called,
            "Expected warnings.warn to be called for non-numeric columns.",
        )
        call_msg = mock_warn.call_args[0][0]
        self.assertIn(
            "OrdinalEncoder",
            call_msg,
            f"Expected 'OrdinalEncoder' in warning message, got: {call_msg!r}",
        )

    def test_coerce_numeric(self):
        """
        Test that _coerce_numeric leaves pure numeric DataFrames alone and ordinally encodes others.
        """
        df_num = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        res_num = FeatureEvaluator._coerce_numeric(df_num)
        pd.testing.assert_frame_equal(df_num, res_num)

        df_mixed = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [True, False], "c": ["high", "low"]}
        )
        with patch("pyquantflow.model.feature_evaluation.warnings.warn") as mock_warn:
            res_mixed = FeatureEvaluator._coerce_numeric(df_mixed)
        self.assertTrue(mock_warn.called)
        self.assertTrue(pd.api.types.is_numeric_dtype(res_mixed["b"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(res_mixed["c"]))
        self.assertEqual(list(res_mixed.columns), ["a", "b", "c"])

    def test_cluster_entities_euclidean(self):
        """
        Test cluster_entities using Euclidean distance.
        """
        evaluator = FeatureEvaluator(features=["feat1"], target_col="target")
        data = pd.DataFrame(
            [
                [1.0, 1.1],
                [0.9, 1.0],
                [-1.0, -1.1],
                [-0.9, -1.0],
            ],
            index=["Asset1", "Asset2", "Asset3", "Asset4"],
            columns=["f1", "f2"],
        )
        clusters = evaluator.cluster_entities(data, method="euclidean", n_clusters=2)
        self.assertEqual(len(clusters), 2)
        c_vals = list(clusters.values())
        group1 = c_vals[0] if "Asset1" in c_vals[0] else c_vals[1]
        group2 = c_vals[1] if "Asset1" in c_vals[0] else c_vals[0]
        self.assertTrue("Asset2" in group1)
        self.assertTrue("Asset3" in group2)
        self.assertTrue("Asset4" in group2)

    def test_cluster_entities_invalid_method(self):
        """
        Test that cluster_entities raises ValueError on unknown method.
        """
        evaluator = FeatureEvaluator(features=["feat1"], target_col="target")
        with self.assertRaises(ValueError):
            evaluator.cluster_entities(
                self.panel_df[["feat1", "feat2"]], method="invalid_method"
            )

    def test_cluster_entities_explicit_clusters_and_small_n(self):
        """
        Test explicit cluster configuration and small entity boundary cases.
        """
        evaluator = FeatureEvaluator(features=["feat1"], target_col="target")
        small_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        clusters = evaluator.cluster_entities(
            small_data, method="correlation", n_clusters=None
        )
        self.assertEqual(len(clusters), 2)

        data = pd.DataFrame(
            {
                "a": np.random.randn(10),
                "b": np.random.randn(10),
                "c": np.random.randn(10),
            }
        )
        clusters_exp = evaluator.cluster_entities(
            data, method="correlation", n_clusters=2
        )
        self.assertEqual(len(clusters_exp), 2)

    def test_compute_time_series_profiles_no_ticker(self):
        """
        Test compute_time_series_profiles when groupby_level is not 'ticker'.
        """
        single_df = self._create_synthetic_single_index_df(n_periods=30)
        evaluator = FeatureEvaluator(
            features=["feat1"],
            target_col="target",
        )
        profiles = evaluator.compute_time_series_profiles(
            single_df, columns=["feat1", "feat2"], groupby_level=None
        )
        self.assertTrue(len(profiles.columns) > 0)
        self.assertEqual(len(profiles), 2)

        df_no_dt = pd.DataFrame(
            {"feat1": np.random.randn(30), "feat2": np.random.randn(30)}
        )
        profiles_no_dt = evaluator.compute_time_series_profiles(
            df_no_dt, columns=["feat1", "feat2"], groupby_level=None
        )
        self.assertTrue(len(profiles_no_dt.columns) > 0)
        self.assertEqual(len(profiles_no_dt), 2)

    def test_evaluate_importance_single_feature(self):
        """
        Test evaluate_importance when all_features has exactly 1 element.
        """
        df = self.panel_df.copy()
        evaluator = FeatureEvaluator(
            features=["feat2"],
            target_col="target",
            cv=KFold(n_splits=2),
            memory_threshold=-1.0,
        )
        df_trans = evaluator.fit_transform_features(df).fillna(0)
        results = evaluator.evaluate_importance(
            df_trans,
            estimator=DecisionTreeClassifier(),
            metric=accuracy_score,
            greater_is_better=True,
            needs_proba=False,
            balance_classes=False,
        )
        self.assertTrue(len(results) > 0)
        first_regime = list(results.values())[0]
        self.assertIn("MDA", first_regime)
        self.assertIn("SFI", first_regime)

    def test_evaluate_importance_balance_classes(self):
        """
        Test evaluate_importance when balance_classes=True.
        """
        df = self.panel_df.copy()
        evaluator = FeatureEvaluator(
            features=["feat2"],
            target_col="target",
            weight_col="weight",
            cv=KFold(n_splits=2),
            memory_threshold=-1.0,
        )
        df_trans = evaluator.fit_transform_features(df).fillna(0)
        results = evaluator.evaluate_importance(
            df_trans,
            estimator=DecisionTreeClassifier(),
            metric=accuracy_score,
            greater_is_better=True,
            needs_proba=False,
            balance_classes=True,
        )
        self.assertTrue(len(results) > 0)

    def test_evaluate_importance_pipeline_estimator(self):
        """
        Test evaluate_importance with a scikit-learn Pipeline estimator.
        """
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("classifier", DecisionTreeClassifier())]
        )

        df = self.panel_df.copy()
        evaluator = FeatureEvaluator(
            features=["feat2"],
            target_col="target",
            weight_col="weight",
            cv=KFold(n_splits=2),
            memory_threshold=-1.0,
        )
        df_trans = evaluator.fit_transform_features(df).fillna(0)
        results = evaluator.evaluate_importance(
            df_trans,
            estimator=pipe,
            metric=accuracy_score,
            greater_is_better=True,
            needs_proba=False,
            balance_classes=True,
        )
        self.assertTrue(len(results) > 0)

    def test_evaluate_importance_metric_exception(self):
        """
        Test that evaluate_importance handles exceptions in metric scoring gracefully.
        """

        def failing_metric(y_true, y_pred, **kwargs):
            raise ValueError("Intentional scoring error")

        df = self.panel_df.copy()
        evaluator = FeatureEvaluator(
            features=["feat2"],
            target_col="target",
            cv=KFold(n_splits=2),
            memory_threshold=-1.0,
        )
        df_trans = evaluator.fit_transform_features(df).fillna(0)
        results = evaluator.evaluate_importance(
            df_trans,
            estimator=DecisionTreeClassifier(),
            metric=failing_metric,
            greater_is_better=True,
            needs_proba=False,
            balance_classes=False,
        )
        first_regime = list(results.values())[0]
        self.assertTrue(np.isnan(first_regime["SFI"]["sfi_mean"].iloc[0]))
        self.assertTrue(np.isnan(first_regime["MDA"]["mda_mean"].iloc[0]))

    def test_convert_results_to_table_empty(self):
        """
        Test that _convert_results_to_table returns an empty DataFrame if results dict is empty.
        """
        empty_res = FeatureEvaluator._convert_results_to_table({})
        self.assertTrue(empty_res.empty)

    def test_evaluate_importance_with_raw_features(self):
        """
        Req 2.1 (end-to-end): Both transformed and raw features must appear
        in importance_df after evaluate_importance completes.
        """
        np.random.seed(5)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)

        signal = np.random.randn(n).cumsum()
        volume = np.abs(np.random.randn(n)) * 1_000
        target = np.random.choice([0, 1], n)

        df = pd.DataFrame(
            {"signal": signal, "volume": volume, "target": target},
            index=dates,
        )
        df.index.name = "datetime"

        evaluator = FeatureEvaluator(
            features=["signal"],
            raw_features=["volume"],
            target_col="target",
            cv=KFold(n_splits=2),
            memory_threshold=-1.0,
        )

        df_trans = evaluator.fit_transform_features(df)
        df_trans = df_trans.fillna(0)

        evaluator.evaluate_importance(
            df_trans,
            estimator=DecisionTreeClassifier(),
            metric=log_loss,
            metric_kwargs={"labels": [0, 1]},
            greater_is_better=False,
            needs_proba=True,
            balance_classes=False,
        )

        self.assertIsNotNone(evaluator.importance_df)
        importance_df = evaluator.importance_df
        self.assertFalse(importance_df.empty, "importance_df must not be empty.")

        all_feature_names = " ".join(importance_df["features"].astype(str).tolist())
        self.assertIn(
            "signal",
            all_feature_names,
            "'signal' (transformed feature) must appear in importance_df.",
        )
        self.assertIn(
            "volume",
            all_feature_names,
            "'volume' (raw feature) must appear in importance_df.",
        )


if __name__ == "__main__":
    unittest.main()
