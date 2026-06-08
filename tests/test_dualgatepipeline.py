import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.model.feature_evaluation import FeatureEvaluator
from pyquantflow.data.features.dualgatepipeline import DualGatePipelineFactory
from pyquantflow.data.labels.factory import TrendScanningLabelFactory


class TestDualGatePipeline(unittest.TestCase):
    """Tests verifying DualGatePipelineFactory orchestration and alignment logic."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range(
            start="2023-01-01", periods=n, freq="D", tz="Australia/Sydney"
        )

        self.clean_daily_map = {}
        for ticker in ["AAPL", "MSFT"]:
            # Generate random walk close prices
            returns = np.random.normal(0.0005, 0.01, n)
            close = 100 * np.cumprod(1 + returns)
            high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
            low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
            open_ = (high + low) / 2
            volume = np.random.randint(1000, 100000, n)

            df = pd.DataFrame(
                {
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "feat1": np.random.randn(n).cumsum(),  # non-stationary
                    "feat2": np.random.randn(n),  # stationary
                },
                index=dates,
            )
            df.index.name = "datetime"
            self.clean_daily_map[ticker] = df

    def test_dual_gate_pipeline_execution(self):
        """Verify the pipeline executes without timezone, comparison or index alignment errors."""
        label_factory = TrendScanningLabelFactory(
            windows=[5, 10, 20],
            bins=[-2.0, 2.0],
        )

        ao = AssetOrganiser(
            data_map=self.clean_daily_map,
            cutoff_date="2023-07-01",
            target_features=["label"],
            weight_col="weight",
            label_factory=label_factory,
        )

        evaluator = FeatureEvaluator(
            features=["feat1", "feat2"],
            target_col="label",
            weight_col="weight",
            t1_col="t1",
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            memory_threshold=-1.0,  # Keep all features to avoid accidental empty feature lists
        )

        factory = DualGatePipelineFactory(
            price_col="Close",
            filter_col="Close",
        )

        # Execute the pipeline with target_events_train=150
        ao, active_features = factory.execute(
            organiser=ao,
            evaluator=evaluator,
            target_events_train=150,
            target_labels=["label", "t1", "weight"],
        )

        # Assertions to verify the organiser contains the processed and transformed data
        self.assertIsNotNone(ao)
        self.assertIsNotNone(ao.multi_asset)
        self.assertEqual(ao.multi_asset.index.names, ["datetime", "ticker"])
        self.assertEqual(
            str(ao.multi_asset.index.get_level_values("datetime").tz),
            "Australia/Sydney",
        )

        # The organiser should have generated labels, t1 and weight columns
        self.assertIn("label", ao.multi_asset.columns)
        self.assertIn("t1", ao.multi_asset.columns)
        self.assertIn("weight", ao.multi_asset.columns)

        # Verified features should still exist in importance_df
        self.assertIsNotNone(active_features)

    def test_dual_gate_pipeline_with_precomputed_indicators(self):
        """Verify the pipeline executes correctly when FFD and SADF are pre-computed on the input data map."""
        from pyquantflow.data.features.indicator import FRACTIONAL_DIFF, SADF_JAX

        # Add FFD and SADF to each ticker's DataFrame in clean_daily_map
        for ticker, df in self.clean_daily_map.items():
            # Apply indicators on Close price Series with looser thres to preserve enough rows
            df["ffd_close"] = FRACTIONAL_DIFF(df["Close"], d=0.4, thres=1e-2)
            df["sadf_close"] = SADF_JAX(df["Close"], min_length=20)

        label_factory = TrendScanningLabelFactory(
            windows=[5, 10, 20],
            bins=[-2.0, 2.0],
        )

        ao = AssetOrganiser(
            data_map=self.clean_daily_map,
            cutoff_date="2023-07-01",
            target_features=["label"],
            weight_col="weight",
            label_factory=label_factory,
        )

        # Include our pre-computed indicators in the feature evaluator feature list
        evaluator = FeatureEvaluator(
            features=["feat1", "feat2", "ffd_close", "sadf_close"],
            target_col="label",
            weight_col="weight",
            t1_col="t1",
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            memory_threshold=-1.0,  # Keep all features to avoid dropping
        )

        factory = DualGatePipelineFactory(
            price_col="Close",
            filter_col="Close",
        )

        # Execute the pipeline with target_events_train=150
        ao, active_features = factory.execute(
            organiser=ao,
            evaluator=evaluator,
            target_events_train=150,
            target_labels=["label", "t1", "weight"],
        )

        # Assertions to verify the organiser contains the processed and transformed data
        self.assertIsNotNone(ao)
        self.assertIsNotNone(ao.multi_asset)
        self.assertEqual(ao.multi_asset.index.names, ["datetime", "ticker"])
        self.assertEqual(
            str(ao.multi_asset.index.get_level_values("datetime").tz),
            "Australia/Sydney",
        )

        # The organiser should have generated labels, t1 and weight columns
        self.assertIn("label", ao.multi_asset.columns)
        self.assertIn("t1", ao.multi_asset.columns)
        self.assertIn("weight", ao.multi_asset.columns)

        # The pre-computed columns should exist in the multi_asset DataFrame
        self.assertIn("ffd_close", ao.multi_asset.columns)
        self.assertIn("sadf_close", ao.multi_asset.columns)


if __name__ == "__main__":
    unittest.main()
