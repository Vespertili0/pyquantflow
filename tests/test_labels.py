import unittest
import pandas as pd
import numpy as np
from pyquantflow.data.labels.triple_barrier import apply_triple_barrier
from pyquantflow.data.labels.sample_weights import get_sample_weights
from pyquantflow.data.labels import TrendScanningLabelFactory


class TestLabelsAndWeights(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range("2021-04-28", periods=100, tz="UTC")
        self.prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=self.dates)
        self.sl_col = self.prices * 0.98

    def test_triple_barrier_and_sample_weights(self):
        # 1. Test Triple Barrier
        barrier_data = apply_triple_barrier(
            self.prices, self.sl_col, tp_mult=3, horizon=10
        )
        self.assertIsInstance(barrier_data, pd.DataFrame)
        self.assertIn("t1", barrier_data.columns)
        self.assertIn("label", barrier_data.columns)

        # t1 dtype should explicitly match index dtype or be a proper datetime array to avoid warnings
        t1 = barrier_data["t1"]
        returns = self.prices.pct_change()

        # 2. Test Sample Weights
        # It should handle t1 containing NaTs from Triple Barrier without throwing TypeError
        weights = get_sample_weights(t1, returns=returns)

        self.assertIsInstance(weights, pd.Series)
        # Expected weights should have length of non-NaT t1 items at the very least,
        # though get_sample_weights reindexes to t1.index. Let's verify weights output shape matches.
        self.assertEqual(len(weights), len(t1.dropna()))

        # The result should not be entirely NaNs
        self.assertFalse(weights.isna().all())

    def test_trend_scanning_label_factory(self):
        # Create a mock ticker DataFrame
        ticker_df = pd.DataFrame({"Close": self.prices}, index=self.dates)

        # 1. Test with default bins [-10, 12]
        factory = TrendScanningLabelFactory(windows=[5, 10], bins=[-10.0, 12.0])
        labels_df = factory.generate_labels(ticker_df, price_col="Close")

        self.assertIn("label", labels_df.columns)
        self.assertIn("t_value", labels_df.columns)
        self.assertIn("t1", labels_df.columns)

        # Verify that classes are generated and valid values belong to {0.0, 1.0, 2.0}
        valid_labels = labels_df["label"].dropna()
        self.assertTrue(len(valid_labels) > 0)
        self.assertTrue(set(valid_labels.unique()).issubset({0.0, 1.0, 2.0}))

        # 2. Test with custom bins (e.g. [0.0] for binary mapping if user needs it, or custom ternary)
        factory_custom = TrendScanningLabelFactory(windows=[5], bins=[0.0])
        labels_custom = factory_custom.generate_labels(ticker_df, price_col="Close")
        valid_custom = labels_custom["label"].dropna()
        self.assertTrue(set(valid_custom.unique()).issubset({0.0, 1.0}))

    def test_asset_organiser_label_dropping(self):
        # Verify that AssetOrganiser drops rows with NaN/NaT values in label-related columns
        from pyquantflow.data.assetorganiser import AssetOrganiser
        from pyquantflow.data.labels import TripleBarrierLabelFactory

        # Create mock data map with 100 days of price data
        df = pd.DataFrame(
            {
                "Close": self.prices,
                "feature1": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            },
            index=self.dates,
        )
        df.index.name = "datetime"
        data_map = {"AAA": df}

        factory = TripleBarrierLabelFactory(pt_mult=1.0, sl_mult=1.0, horizon=5)
        organiser = AssetOrganiser(
            data_map=data_map,
            cutoff_date="2021-06-01",
            target_features=["target"],
            label_factory=factory,
        )
        organiser.prepare_multi_asset_frame()

        # Before applying labels, multi_asset has 100 rows
        initial_len = len(organiser.multi_asset)
        self.assertEqual(initial_len, 100)

        # Apply continuous labels (which should drop rows with NaN values in the label-related columns)
        organiser.apply_continuous_labels(price_col="Close")

        # The length of multi_asset should be less than 100 because the most recent datapoints
        # (near the end of the series within the horizon) will have NaN values and thus be dropped
        new_len = len(organiser.multi_asset)
        self.assertTrue(new_len < 100)

        # Ensure no NaN values exist in the label-related columns
        self.assertFalse(organiser.multi_asset["label"].isna().any())
        self.assertFalse(organiser.multi_asset["t1"].isna().any())

    def test_asset_organiser_weight_dropping(self):
        # Verify that AssetOrganiser drops rows with NaN values in weights column
        from pyquantflow.data.assetorganiser import AssetOrganiser
        from pyquantflow.data.labels import TripleBarrierLabelFactory

        # Create mock data map with 100 days of price data
        df = pd.DataFrame(
            {
                "Close": self.prices,
                "feature1": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            },
            index=self.dates,
        )
        df.index.name = "datetime"
        data_map = {"AAA": df}

        factory = TripleBarrierLabelFactory(pt_mult=1.0, sl_mult=1.0, horizon=5)
        organiser = AssetOrganiser(
            data_map=data_map,
            cutoff_date="2021-06-01",
            target_features=["target"],
            label_factory=factory,
            weight_col="my_weight",
        )
        organiser.prepare_multi_asset_frame()
        organiser.apply_continuous_labels(price_col="Close")

        # Now apply sample weights
        # Note: the first element of the returns will be NaN (due to pct_change)
        # resulting in a NaN weight. This first row should be dropped.
        len_before_weights = len(organiser.multi_asset)
        organiser.apply_sample_weights(price_col="Close")
        len_after_weights = len(organiser.multi_asset)

        # Ensure that the row with NaN weight was dropped
        self.assertEqual(len_after_weights, len_before_weights - 1)
        self.assertFalse(organiser.multi_asset["my_weight"].isna().any())


if __name__ == "__main__":
    unittest.main()
