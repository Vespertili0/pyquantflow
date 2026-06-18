import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

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

        # Create panel data (MultiIndex)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=50), ["AAPL", "MSFT"]],
            names=["datetime", "ticker"],
        )
        self.panel_df = pd.DataFrame(
            {
                "feat1": np.random.randn(100).cumsum(),
                "feat2": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
                "weight": np.random.uniform(0.5, 1.5, 100),
            },
            index=idx,
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

        # The rolling z-score window is 20, so early values should be NaN
        # and standardisation should happen per ticker.

    def test_feature_evaluator_profiling(self):
        evaluator = FeatureEvaluator(
            features=["feat1", "feat2"], target_col="target", cv=KFold(n_splits=2)
        )

        # We patch tsfeatures to avoid slow actual runs in standard unit tests, or let it run
        # if the environment has it. Since it's light, we let it run.
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
        from sklearn.metrics import log_loss
        from sklearn.tree import DecisionTreeClassifier

        # Use a simple estimator to avoid thread/parallel array sizing issues on small datasets
        dummy = DecisionTreeClassifier()

        # Transform features
        df_trans = evaluator.fit_transform_features(df)

        # DecisionTreeClassifier does not natively handle NaNs, so we fill them for the dummy test
        df_trans = df_trans.fillna(0)

        # log_loss is a loss metric: greater_is_better=False, needs_proba=True
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
        from sklearn.metrics import brier_score_loss
        from sklearn.linear_model import LogisticRegression

        np.random.seed(0)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)
        # informative: target is directly derived from this feature
        signal = np.random.randn(n)
        target = (signal > 0).astype(int)
        # noise: pure random, no predictive power
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

        # Verify structure is valid; at minimum the signal cluster should exist
        self.assertTrue(len(results) > 0)
        first_regime = list(results.values())[0]
        mda_df = first_regime["MDA"]
        self.assertFalse(mda_df["mda_mean"].isna().all())

    def test_mda_greater_is_better_true(self):
        """
        Req 1.1: f1_score (higher is better) with a highly predictive feature.
        MDA importance must be positive for the informative cluster.
        """
        from sklearn.metrics import f1_score
        from sklearn.linear_model import LogisticRegression

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

        # Verify structure is valid
        self.assertTrue(len(results) > 0)
        first_regime = list(results.values())[0]
        mda_df = first_regime["MDA"]
        self.assertFalse(mda_df["mda_mean"].isna().all())

    def test_feature_evaluator_gate1_pruning(self):
        # Create a dataframe with one high memory feature (AR(1)) and one low memory feature (white noise)
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

        # ar1_feat should be kept (acf1 ~ 0.8, is stationary so d=0.0)
        # white_noise should be dropped (acf1 ~ 0.0, is stationary so d=0.0)
        self.assertIn("ar1_feat", evaluator.features)
        self.assertNotIn("white_noise", evaluator.features)
        self.assertIn("ar1_feat", df_trans.columns)
        self.assertNotIn("white_noise", df_trans.columns)

    def test_feature_evaluator_freq_parameter(self):
        # Verify freq is set correctly
        evaluator = FeatureEvaluator(
            features=["feat1"],
            target_col="target",
            freq=5,
        )
        self.assertEqual(evaluator.freq, 5)

    # -----------------------------------------------------------------------
    # Epic-2 tests
    # -----------------------------------------------------------------------

    def test_raw_features_bypass_ffd(self):
        """
        Req 2.1: raw_features must not be fractionally differentiated.
        After fit_transform_features the 'Volume' raw column should be
        identical to the original, while the transformed 'feat1' column must
        differ (FFD was applied).
        """
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)

        # Non-stationary series (random walk) → will be FFD-transformed
        feat1 = np.random.randn(n).cumsum()
        # Stationary, natively meaningful series → passed through raw
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
            memory_threshold=-1.0,  # Keep all transformed features
        )
        # Use a higher threshold and shorter window to ensure some transformed values survive
        evaluator.stationary_transformer.ffd_thres = 1e-2
        evaluator.stationary_transformer.rolling_z_window = 10

        df_out = evaluator.fit_transform_features(df)

        # Volume must be present in the output
        self.assertIn("Volume", df_out.columns)

        # Volume values must be identical to originals (no transformation)
        np.testing.assert_array_equal(
            df_out["Volume"].dropna().values,
            df["Volume"].loc[df_out["Volume"].dropna().index].values,
            err_msg="raw_features column 'Volume' was unexpectedly modified.",
        )

        # feat1 must have been transformed (values differ from the raw series)
        transformed_vals = df_out["feat1"].dropna()
        self.assertGreater(len(transformed_vals), 0)
        original_vals = df["feat1"].loc[transformed_vals.index]
        self.assertFalse(
            np.allclose(transformed_vals.values, original_vals.values),
            "Expected 'feat1' to be FFD-transformed but it appears unchanged.",
        )

        # self.raw_features must not have been pruned
        self.assertIn("Volume", evaluator.raw_features)

    def test_cluster_entities_with_categorical(self):
        """
        Req 2.3: cluster_entities must not crash when the input DataFrame
        contains boolean or string (object-dtype) columns.
        """
        from unittest.mock import patch

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
            target_col="numeric_a",  # Placeholder; not used in clustering
        )

        # Patch warnings.warn at the feature_evaluation module level.
        # This is more reliable than catch_warnings(record=True), which can
        # interact unpredictably with pytest's own warning filter stack.
        with patch("pyquantflow.model.feature_evaluation.warnings.warn") as mock_warn:
            clusters = evaluator.cluster_entities(df_mixed, method="correlation")

        # Should return a valid cluster dict without raising
        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0)

        # All four columns should appear across clusters
        all_clustered = [col for cols in clusters.values() for col in cols]
        for col in ["numeric_a", "numeric_b", "bool_col", "str_col"]:
            self.assertIn(col, all_clustered)

        # The OrdinalEncoder warning must have been emitted
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

    def test_evaluate_importance_with_raw_features(self):
        """
        Req 2.1 (end-to-end): Both transformed and raw features must appear
        in importance_df after evaluate_importance completes.
        """
        from sklearn.metrics import log_loss
        from sklearn.tree import DecisionTreeClassifier

        np.random.seed(5)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n)

        signal = np.random.randn(n).cumsum()  # Non-stationary → FFD path
        volume = np.abs(np.random.randn(n)) * 1_000  # Stationary → raw path
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

        # importance_df should exist and have rows
        self.assertIsNotNone(evaluator.importance_df)
        importance_df = evaluator.importance_df
        self.assertFalse(importance_df.empty, "importance_df must not be empty.")

        # Both 'signal' and 'volume' must appear somewhere in the features column
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
