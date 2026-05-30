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


if __name__ == "__main__":
    unittest.main()
