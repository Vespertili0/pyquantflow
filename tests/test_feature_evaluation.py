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

        results = evaluator.evaluate_importance(
            df_trans, estimator=dummy, metric=log_loss, balance_classes=False
        )

        # Check regime output structure
        self.assertTrue(len(results.keys()) > 0)
        first_regime = list(results.keys())[0]
        self.assertIn("MDA", results[first_regime])
        self.assertIn("SFI", results[first_regime])
        self.assertTrue(len(results[first_regime]["MDA"]) > 0)

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


if __name__ == "__main__":
    unittest.main()
