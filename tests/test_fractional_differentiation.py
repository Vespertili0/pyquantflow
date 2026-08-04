import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import os
from pyquantflow.data.database import DatabaseManager
from pyquantflow.data.features.fractional_differentiation import (
    frac_diff_ffd,
    adf_screened_ffd,
    _adf_test_stat,
    _adf_p_value,
)


class TestFractionalDifferentiation(unittest.TestCase):
    """Unit tests for fractional differentiation functions and ADF screening."""

    def setUp(self):
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.ohlc_data = None

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                for ticker in ["FMG.AX", "CBA.AX"]:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 100:
                        self.ohlc_data = df
                        break
                db_manager.conn.close()
            except Exception:
                pass

        if self.ohlc_data is None:
            self.ohlc_data = self.generate_synthetic_ohlc()

    def generate_synthetic_ohlc(self, n=500, seed=42):
        """Generates synthetic OHLC data as fallback."""
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
        returns = np.random.normal(0, 0.01, n)
        price_path = 100 * np.cumprod(1 + returns)
        high = price_path * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = price_path * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = (high + low) / 2 + np.random.normal(0, 0.002, n)
        high = np.maximum(high, np.maximum(open_, price_path))
        low = np.minimum(low, np.minimum(open_, price_path))
        volume = np.random.randint(1000, 100000, n)
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": price_path,
                "Volume": volume,
            },
            index=dates,
        )
        return df

    def test_adf_test_stat_basic(self):
        """Test basic ADF t-statistic computation."""
        series = self.ohlc_data["Close"]
        t_stat = _adf_test_stat(series, lags=1)
        self.assertIsInstance(t_stat, float)
        self.assertFalse(np.isnan(t_stat))

    def test_adf_test_stat_short_series(self):
        """Test that ADF t-statistic returns NaN for series shorter than lags + 2."""
        short_series = pd.Series([10.0, 11.0])
        # lags = 1, required length is lags + 2 = 3. 2 < 3, so should return NaN
        t_stat = _adf_test_stat(short_series, lags=1)
        self.assertTrue(np.isnan(t_stat))

    def test_adf_test_stat_linalg_error_inv(self):
        """Test that ADF t-statistic returns NaN when matrix inversion raises LinAlgError."""
        # A perfectly collinear series (constant difference) will result in a singular X.T @ X matrix
        # dy will be [1.0, 1.0, ...], constant column is [1.0, 1.0, ...], making them perfectly collinear
        collinear_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # When lags=1, X will have [y_lag, constant, dy_lag]
        t_stat = _adf_test_stat(collinear_series, lags=1)

        # Should return NaN due to LinAlgError during np.linalg.inv
        self.assertTrue(np.isnan(t_stat))

    @patch("numpy.linalg.lstsq")
    def test_adf_test_stat_linalg_error_lstsq(self, mock_lstsq):
        """Test that ADF t-statistic returns NaN when np.linalg.lstsq raises LinAlgError."""
        mock_lstsq.side_effect = np.linalg.LinAlgError(
            "SVD did not converge in Linear Least Squares"
        )
        series = pd.Series([1.0, 2.5, 3.1, 4.8, 5.2, 6.9])

        t_stat = _adf_test_stat(series, lags=1)

        self.assertTrue(np.isnan(t_stat))
        mock_lstsq.assert_called_once()

    def test_adf_p_value_thresholds(self):
        """Test ADF p-value approximations at MacKinnon boundary thresholds."""
        # 1. Below 1% critical value (-3.43)
        self.assertEqual(_adf_p_value(-3.50), 0.01)
        # 2. Exactly 1% critical value (-3.43)
        self.assertEqual(_adf_p_value(-3.43), 0.01)
        # 3. Between 1% and 5% (-3.43 to -2.86)
        # Interpolation test (midpoint should yield 0.03 exactly)
        self.assertAlmostEqual(_adf_p_value(-3.145), 0.03)
        # 4. Between 5% and 10% (-2.86 to -2.57)
        # Interpolation test (midpoint should yield 0.075 exactly)
        self.assertAlmostEqual(_adf_p_value(-2.715), 0.075)
        # 5. Above 10% critical value (-2.57)
        self.assertEqual(_adf_p_value(-2.0), 1.0)
        self.assertEqual(_adf_p_value(0.0), 1.0)
        # 6. NaN handling
        self.assertEqual(_adf_p_value(np.nan), 1.0)

    def test_frac_diff_ffd_properties(self):
        """Test properties of fractional_differentiation frac_diff_ffd function."""
        series = self.ohlc_data["Close"]
        d = 0.4
        result = frac_diff_ffd(series, d=d, thres=1e-4)

        # Output properties
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(series))
        self.assertTrue(result.index.equals(series.index))

        # Output name formatting
        self.assertEqual(result.name, f"frac_diff_{d}")

        # Check NaN padding warm-up period
        # First few elements should be NaN
        self.assertTrue(np.isnan(result.iloc[0]))
        # Last elements should be valid numbers
        self.assertFalse(np.isnan(result.iloc[-1]))

    def test_frac_diff_ffd_first_differencing(self):
        """Test that d=1.0 is equivalent to first differencing with NaN at start."""
        series = self.ohlc_data["Close"]
        result = frac_diff_ffd(series, d=1.0, thres=1e-4)

        # Hand-calculated first difference
        expected_diff = series.diff()

        # Compare starting from index 1
        np.testing.assert_array_almost_equal(
            result.iloc[1:].values, expected_diff.iloc[1:].values
        )
        self.assertTrue(np.isnan(result.iloc[0]))

    def test_adf_screened_ffd_explicit_mode(self):
        """Test adf_screened_ffd in explicit mode when d is provided."""
        series = self.ohlc_data["Close"]
        d = 0.5

        # Test with pd.Series
        result_series, d_used_series = adf_screened_ffd(series, d=d)
        self.assertEqual(d_used_series, d)
        self.assertIsInstance(result_series, pd.Series)
        self.assertTrue(result_series.index.equals(series.index))

        # Test with np.ndarray
        arr = series.values
        result_arr, d_used_arr = adf_screened_ffd(arr, d=d)
        self.assertEqual(d_used_arr, d)
        self.assertIsInstance(result_arr, pd.Series)
        self.assertEqual(len(result_arr), len(arr))

    def test_adf_screened_ffd_screening_mode(self):
        """Test adf_screened_ffd in screening mode to find optimal d."""
        series = self.ohlc_data["Close"]

        # Running screening with d=None
        result, d_star = adf_screened_ffd(series, d=None)

        self.assertGreater(d_star, 0.0)
        self.assertLessEqual(d_star, 1.0)

        # Check that the resulting series achieves stationarity
        t_stat = _adf_test_stat(result)
        p_val = _adf_p_value(t_stat)
        self.assertLessEqual(p_val, 0.05)

    def test_adf_screened_ffd_fallback(self):
        """Test adf_screened_ffd fallback behaviour when no stationarity is achieved."""
        # Create an extremely non-stationary, explosive series
        n = 100
        explosive_series = pd.Series(np.exp(np.arange(n) * 0.2))

        # Use a very strict significance level and grid excluding 1.0
        result, d_star = adf_screened_ffd(
            explosive_series,
            d=None,
            significance_level=1e-8,
            d_grid=np.arange(0.25, 0.95, 0.05),
        )

        # It must fallback to d=1.0 since none in the grid can achieve stationarity
        self.assertEqual(d_star, 1.0)

        # Compare output to first difference logic
        expected = frac_diff_ffd(explosive_series, d=1.0)
        np.testing.assert_array_almost_equal(result.values, expected.values)


if __name__ == "__main__":
    unittest.main()
