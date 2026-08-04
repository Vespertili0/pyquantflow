"""
Unit tests for the Market Microstructure feature estimators.

Tests construct synthetic price series with known, analytically deterministic
bid-ask spreads so the estimators can be verified for approximate recovery.
"""

import unittest

import numpy as np

from pyquantflow.data.features.microstructure import CORWIN_SCHULTZ, ROLL_MEASURE


class TestRollMeasure(unittest.TestCase):
    """Tests for the Roll (1984) effective spread estimator."""

    def _alternating_series(self, n: int, mid: float, half_spread: float) -> np.ndarray:
        """
        Constructs a price series that *perfectly* alternates between
        ``mid + half_spread`` and ``mid - half_spread``.

        Used only for structural tests (NaN padding, length checks).
        Do NOT use for spread-value recovery tests: a perfectly alternating
        series violates Roll's iid ``q_t`` assumption (``q_t`` alternates
        deterministically), causing the serial covariance to equal ``-s²``
        rather than ``-(s/2)²`` and producing a 2× overestimate.
        """
        prices = np.empty(n)
        for i in range(n):
            prices[i] = mid + half_spread if i % 2 == 0 else mid - half_spread
        return prices

    def _stochastic_bid_ask_series(
        self, n: int, mid: float, half_spread: float, seed: int = 42
    ) -> np.ndarray:
        """
        Constructs a price series with random (iid) bid-ask bounce:

        .. code-block:: text

            P_t = mid + half_spread * q_t,   q_t \u223c iid Uniform({-1, +1})

        Under Roll's model (iid ``q_t``):

        .. math::

            \\text{Cov}(\\Delta P_t, \\Delta P_{t-1})
            = -\\text{half\\_spread}^2

        so Roll's formula recovers
        ``spread = 2 * half_spread`` exactly in expectation.
        """
        rng = np.random.default_rng(seed)
        q = rng.choice([-1, 1], n)
        return (mid + half_spread * q).astype(np.float64)

    def test_deterministic_small_array(self):
        """Test exact expected values on a small known array from the docstring."""
        close = np.array([100.0, 100.5, 99.5, 100.5, 99.5, 100.5])
        result = ROLL_MEASURE(close, window=3)
        expected = np.array([np.nan, np.nan, np.nan, np.nan, 2.1602469, 2.30940108])
        np.testing.assert_allclose(result, expected, rtol=1e-6, equal_nan=True)

    def test_output_length_matches_input(self):
        """Output array must be the same length as the input."""
        close = np.random.randn(50).cumsum() + 100
        result = ROLL_MEASURE(close, window=10)
        self.assertEqual(len(result), len(close))

    def test_leading_nans_for_cold_start(self):
        """First window+1 values must be NaN (cold-start padding)."""
        close = self._alternating_series(60, mid=100.0, half_spread=0.5)
        window = 15
        result = ROLL_MEASURE(close, window=window)
        nan_count = np.sum(np.isnan(result))
        # At minimum, the first window+1 positions must be NaN
        self.assertGreaterEqual(nan_count, window + 1)
        # Sanity: there should be some non-NaN values after the warm-up
        self.assertTrue(np.any(~np.isnan(result)))

    def test_spread_recovery_on_stochastic_series(self):
        """
        On a long stochastic bid-ask bounce series (iid q_t \u223c \u00b11), Roll's
        estimator should recover approximately ``2 * half_spread``.

        The stochastic series satisfies Roll's model assumption::

            Cov(\u0394P_t, \u0394P_{t-1}) = -(half_spread)\u00b2

        so Roll's formula gives ``spread = 2 * sqrt(half_spread\u00b2) = 2 * half_spread``.

        A 30 % relative tolerance accounts for finite-sample bias in the
        rolling covariance estimator.
        """
        half_spread = 0.25
        true_spread = 2.0 * half_spread
        n = 1000
        window = 100

        close = self._stochastic_bid_ask_series(n, mid=100.0, half_spread=half_spread)
        result = ROLL_MEASURE(close, window=window)

        # Evaluate on the non-NaN tail only
        valid = result[~np.isnan(result)]
        self.assertGreater(len(valid), 0, "No valid (non-NaN) outputs produced.")

        estimated_spread = np.nanmedian(valid)
        rel_error = abs(estimated_spread - true_spread) / true_spread
        self.assertLess(
            rel_error,
            0.30,
            f"Roll spread recovery error too large: estimated={estimated_spread:.4f}, "
            f"true={true_spread:.4f}, rel_error={rel_error:.2%}",
        )

    def test_nonnegative_outputs(self):
        """Roll spread estimates must be non-negative (clamped at 0)."""
        np.random.seed(7)
        close = np.random.randn(200).cumsum() + 100
        result = ROLL_MEASURE(close, window=20)
        valid = result[~np.isnan(result)]
        self.assertTrue(
            np.all(valid >= 0),
            f"Found negative Roll Measure values: {valid[valid < 0]}",
        )

    def test_accepts_pd_series(self):
        """Function must accept a pd.Series as well as np.ndarray."""
        import pandas as pd

        close = pd.Series(np.random.randn(50).cumsum() + 100)
        result = ROLL_MEASURE(close, window=10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(close))

    def test_invalid_window_raises(self):
        """Window < 2 must raise a ValueError."""
        close = np.random.randn(30).cumsum() + 100
        with self.assertRaises(ValueError):
            ROLL_MEASURE(close, window=1)

    def test_short_series_returns_all_nans(self):
        """Series shorter than window+1 must return all NaN."""
        close = np.array([100.0, 101.0, 99.0])
        result = ROLL_MEASURE(close, window=10)
        self.assertTrue(np.all(np.isnan(result)))


class TestCorwinSchultz(unittest.TestCase):
    """Tests for the Corwin-Schultz (2012) high-low spread estimator."""

    def _synthetic_ohlc(self, n: int, mid: float, half_spread: float) -> tuple:
        """
        Constructs synthetic high/low arrays where the intraday range is
        entirely driven by the spread:
            H_t = mid * (1 + half_spread)
            L_t = mid * (1 - half_spread)

        This is a highly stylised setup; in practice the range also reflects
        volatility. The resulting spread should be non-trivially positive.
        """
        high = np.full(n, mid * (1.0 + half_spread))
        low = np.full(n, mid * (1.0 - half_spread))
        return high, low

    def test_output_length_matches_input(self):
        """Output array must be the same length as the inputs."""
        n = 80
        high = np.random.uniform(101, 105, n)
        low = np.random.uniform(95, 100, n)
        result = CORWIN_SCHULTZ(high, low, window=10)
        self.assertEqual(len(result), n)

    def test_leading_nans_for_cold_start(self):
        """First window+1 values must be NaN (cold-start padding)."""
        n = 60
        window = 15
        high, low = self._synthetic_ohlc(n, mid=100.0, half_spread=0.005)
        result = CORWIN_SCHULTZ(high, low, window=window)
        nan_count = np.sum(np.isnan(result))
        self.assertGreaterEqual(nan_count, window + 1)
        self.assertTrue(np.any(~np.isnan(result)))

    def test_spread_is_in_unit_interval(self):
        """Corwin-Schultz spread estimates must lie in [0, 1]."""
        np.random.seed(12)
        n = 200
        window = 20
        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.normal(0, 0.5, n))
        low = close - np.abs(np.random.normal(0, 0.5, n))
        # Ensure H > L
        high = np.maximum(high, low + 1e-6)

        result = CORWIN_SCHULTZ(high, low, window=window)
        valid = result[~np.isnan(result)]
        self.assertGreater(len(valid), 0)
        self.assertTrue(
            np.all(valid >= 0) and np.all(valid <= 1),
            f"Corwin-Schultz values outside [0,1]: min={valid.min():.4f}, max={valid.max():.4f}",
        )

    def test_positive_spread_for_synthetic_data(self):
        """Corwin-Schultz should return positive estimates for realistic OHLC data."""
        np.random.seed(99)
        n = 200
        window = 20
        high, low = self._synthetic_ohlc(n, mid=100.0, half_spread=0.01)
        result = CORWIN_SCHULTZ(high, low, window=window)
        valid = result[~np.isnan(result)]
        self.assertGreater(len(valid), 0)
        # At least some estimates should be > 0 for a spread-driven series
        self.assertTrue(
            np.any(valid > 0),
            "Expected positive spread estimates but all were zero.",
        )

    def test_accepts_pd_series(self):
        """Function must accept pd.Series inputs as well as np.ndarray."""
        import pandas as pd

        n = 60
        high = pd.Series(np.random.uniform(101, 105, n))
        low = pd.Series(np.random.uniform(95, 100, n))
        result = CORWIN_SCHULTZ(high, low, window=10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), n)

    def test_mismatched_lengths_raises(self):
        """Mismatched high/low lengths must raise a ValueError."""
        high = np.ones(50)
        low = np.ones(40)
        with self.assertRaises(ValueError):
            CORWIN_SCHULTZ(high, low, window=10)

    def test_non_positive_prices_raises(self):
        """Non-positive prices must raise a ValueError."""
        high = np.array([101.0, 102.0, -1.0, 103.0])
        low = np.array([99.0, 98.0, -2.0, 97.0])
        with self.assertRaises(ValueError):
            CORWIN_SCHULTZ(high, low, window=2)

    def test_short_series_returns_all_nans(self):
        """Series shorter than window+2 must return all NaN."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0, 98.0])
        result = CORWIN_SCHULTZ(high, low, window=10)
        self.assertTrue(np.all(np.isnan(result)))

    def test_exact_spread_calculation(self):
        """Test with small known inputs to ensure deterministic calculation output."""
        high = np.array([101.0, 102.0, 101.5, 103.0, 102.0, 103.5])
        low = np.array([99.0, 98.0, 99.5, 97.0, 98.0, 96.5])
        expected = np.array([np.nan, np.nan, np.nan, np.nan, 0.00259673, 0.00561362])

        result = CORWIN_SCHULTZ(high, low, window=3)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
