import unittest
import numpy as np
import pandas as pd

from pyquantflow.data.features.fractional_differentiation import (
    frac_diff_ffd,
    adf_screened_ffd,
    _adf_test_stat,
    _adf_p_value,
)
from pyquantflow.data.features.indicator import FRACTIONAL_DIFF, SADF_JAX


class TestADFScreenedFFD(unittest.TestCase):
    """Tests for the standalone adf_screened_ffd dual-mode function."""

    def setUp(self):
        np.random.seed(42)
        # Random walk (non-stationary)
        self.random_walk = pd.Series(np.random.randn(500).cumsum())
        # Stationary white noise
        self.stationary = pd.Series(np.random.randn(500))

    def test_explicit_mode_matches_baseline(self):
        """
        API Invariance: calling adf_screened_ffd with a fixed d must produce
        identical output to calling frac_diff_ffd directly.
        """
        d = 0.4
        thres = 1e-4

        baseline = frac_diff_ffd(self.random_walk, d=d, thres=thres)
        result, d_used = adf_screened_ffd(self.random_walk, d=d, thres=thres)

        self.assertEqual(d_used, d)
        np.testing.assert_array_equal(result.values, baseline.values)

    def test_explicit_mode_with_numpy_array(self):
        """Explicit mode accepts raw np.ndarray input."""
        d = 0.5
        arr = self.random_walk.values

        result, d_used = adf_screened_ffd(arr, d=d)

        self.assertEqual(d_used, d)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(arr))

    def test_screening_mode_finds_d_star(self):
        """
        Screening mode should find a d* > 0 for a random walk where
        the ADF test achieves p <= 0.05.
        """
        result, d_star = adf_screened_ffd(self.random_walk, d=None)

        self.assertGreater(d_star, 0.0)
        self.assertLessEqual(d_star, 1.0)
        self.assertEqual(len(result), len(self.random_walk))

        # Verify the result is actually stationary
        t_stat = _adf_test_stat(result)
        p_value = _adf_p_value(t_stat)
        self.assertLessEqual(p_value, 0.05)

    def test_screening_mode_stationary_input(self):
        """
        For an already stationary series, screening should find d* = 0.0
        (or the first d in the grid).
        """
        _, d_star = adf_screened_ffd(
            self.stationary, d=None, d_grid=np.arange(0.0, 1.05, 0.05)
        )
        self.assertEqual(d_star, 0.0)


class TestFRACTIONAL_DIFF(unittest.TestCase):
    """Tests for the FRACTIONAL_DIFF TA-Lib style indicator."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.close_array = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
        self.close_series = pd.Series(
            self.close_array,
            index=pd.date_range("2020-01-01", periods=n),
        )

    def test_returns_ndarray_same_length(self):
        """Output must be an np.ndarray of the same length as input."""
        result = FRACTIONAL_DIFF(self.close_array)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.close_array))

    def test_cold_start_nan_padding(self):
        """Leading cold-start window indices must be np.nan."""
        result = FRACTIONAL_DIFF(self.close_array, d=0.4)

        # First few values should be NaN (FFD kernel warm-up)
        self.assertTrue(np.isnan(result[0]))
        # Later values should be valid
        self.assertFalse(np.all(np.isnan(result[-10:])))

    def test_explicit_mode_bypasses_screening(self):
        """Fixed d should bypass the ADF grid search entirely."""
        result = FRACTIONAL_DIFF(self.close_array, d=0.4)

        # Compare with direct frac_diff_ffd call
        baseline = frac_diff_ffd(pd.Series(self.close_array), d=0.4, thres=1e-4)
        np.testing.assert_array_equal(result, baseline.values)

    def test_accepts_pandas_series(self):
        """Should work with pd.Series input."""
        result = FRACTIONAL_DIFF(self.close_series, d=0.3)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.close_series))

    def test_screening_mode_default(self):
        """Default (d=None) should run screening and return valid output."""
        result = FRACTIONAL_DIFF(self.close_array)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.close_array))
        # Should have some valid values
        self.assertFalse(np.all(np.isnan(result)))


class TestSADF_JAX(unittest.TestCase):
    """Tests for the SADF_JAX TA-Lib style indicator."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        # Generate strictly positive prices for log-transform
        self.close_array = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
        self.close_series = pd.Series(
            self.close_array,
            index=pd.date_range("2020-01-01", periods=n),
        )

    def test_returns_ndarray_same_length(self):
        """Output must be an np.ndarray of the same length as input."""
        result = SADF_JAX(self.close_array)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.close_array))

    def test_cold_start_nan_padding(self):
        """Leading indices before min_length must be np.nan."""
        min_length = 20
        result = SADF_JAX(self.close_array, min_length=min_length)

        # Early indices should be NaN
        self.assertTrue(np.isnan(result[0]))
        # Later values should have valid data
        self.assertFalse(np.isnan(result[-1]))

    def test_accepts_pandas_series(self):
        """Should work with pd.Series input (index alignment)."""
        result = SADF_JAX(self.close_series)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.close_series))
        self.assertFalse(np.isnan(result.iloc[-1]))

    def test_nan_propagation_no_crash(self):
        """Both indicators should handle edge cases without crashing."""
        # Short series — may not achieve stationarity or min_length
        short = np.array([100.0, 101.0, 99.5, 102.0, 100.5])

        # FRACTIONAL_DIFF with explicit d should still work
        result_ffd = FRACTIONAL_DIFF(short, d=0.5)
        self.assertEqual(len(result_ffd), len(short))


if __name__ == "__main__":
    unittest.main()
