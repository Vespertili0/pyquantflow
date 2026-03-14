import unittest
import pandas as pd
import numpy as np
from pyquantflow.data.labels.triple_barrier import apply_triple_barrier
from pyquantflow.data.labels.sample_weights import get_sample_weights

class TestLabelsAndWeights(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range("2021-04-28", periods=100, tz="UTC")
        self.prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=self.dates)
        self.sl_col = self.prices * 0.98

    def test_triple_barrier_and_sample_weights(self):
        # 1. Test Triple Barrier
        barrier_data = apply_triple_barrier(self.prices, self.sl_col, tp_mult=3, horizon=10)
        self.assertIsInstance(barrier_data, pd.DataFrame)
        self.assertIn('t1', barrier_data.columns)
        self.assertIn('label', barrier_data.columns)

        # t1 dtype should explicitly match index dtype or be a proper datetime array to avoid warnings
        t1 = barrier_data['t1']
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

if __name__ == '__main__':
    unittest.main()
