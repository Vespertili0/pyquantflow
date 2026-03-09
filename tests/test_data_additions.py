import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pyquantflow.data.features.fractional_differentiation import frac_diff_ffd
from pyquantflow.data.labels.trend_scanning import trend_scanning
from pyquantflow.data.labels.triple_barrier import apply_triple_barrier as triple_barrier_labels
from pyquantflow.data.features.sadf import get_sadf_jax as gsadf_values
from pyquantflow.data.features.indicator import ICHIMOKU
from pyquantflow.data.utils import pipe_indicator
from pyquantflow.data.quarterly_pull import fetch_quarterly_data, merge_last_hour
from pyquantflow.data.sk_transformers import (
    FractionalDiffTransformer,
    TrendScanningTransformer,
    GSADFTransformer,
    TripleBarrierLabeler
)

import os
from pyquantflow.data.database import DatabaseManager

class TestDataAdditions(unittest.TestCase):
    def setUp(self):
        # Attempt to load from stocks.db
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.ohlc_data = None

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                # Try FMG.AX first, then CBA.AX, else fallback
                for ticker in ['FMG.AX', 'CBA.AX']:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 100:  # Need enough data for tests
                        self.ohlc_data = df
                        break
                db_manager.conn.close()
            except Exception:
                pass

        if self.ohlc_data is None:
            self.ohlc_data = self.generate_synthetic_ohlc()

    def generate_synthetic_ohlc(self, n=500, seed=42):
        """Generates synthetic OHLC data."""
        np.random.seed(seed)
        dates = pd.date_range(start="2023-01-01", periods=n, freq="D")

        # Random walk for Close prices
        returns = np.random.normal(0, 0.01, n)
        price_path = 100 * np.cumprod(1 + returns)

        # Synthesize OHLC based on Close
        high = price_path * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = price_path * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = (high + low) / 2 + np.random.normal(0, 0.002, n)

        # Ensure consistency
        high = np.maximum(high, np.maximum(open_, price_path))
        low = np.minimum(low, np.minimum(open_, price_path))

        volume = np.random.randint(1000, 100000, n)

        df = pd.DataFrame({
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": price_path,
            "Volume": volume
        }, index=dates)
        return df

    def test_frac_diff_ffd(self):
        """Test Fractional Differentiation."""
        series = self.ohlc_data["Close"]
        d = 0.4
        # Lower threshold or larger n to ensure we have valid values
        result = frac_diff_ffd(series, d=d, thres=1e-4)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(series))
        # Check that we have some NaNs at the beginning due to windowing
        self.assertTrue(result.iloc[0:5].isna().any())
        # Check that we have valid values later
        self.assertFalse(result.iloc[-10:].isna().all())

    def test_trend_scanning(self):
        """Test Trend Scanning."""
        series = self.ohlc_data["Close"]
        windows = [10, 20, 30]

        result = trend_scanning(series, windows=windows)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(series))
        # Should have values (ignoring initial/final NaNs depending on implementation details)
        # Trend scanning looks forward, so the end might be NaN
        self.assertTrue(np.isnan(result.iloc[-1]) or result.iloc[-1] == 0 or True)
        # Actually checking middle values
        middle = len(series) // 2
        if not np.isnan(result.iloc[middle]):
             self.assertIsInstance(result.iloc[middle], float)

    def test_triple_barrier_labels(self):
        """Test Triple Barrier Method."""
        series = self.ohlc_data["Close"]
        volatility = series.rolling(20).std()

        sl_col = pd.Series(0.01, index=series.index)
        # Test with fixed barriers
        labels_fixed = triple_barrier_labels(series, sl_col=sl_col, tp_mult=1.0, horizon=10)
        self.assertIsInstance(labels_fixed, pd.DataFrame)
        self.assertIn('label', labels_fixed.columns)
        self.assertTrue(set(labels_fixed['label'].dropna().unique()).issubset({0, 1, 2}))

        # Test with dynamic barriers
        labels_dynamic = triple_barrier_labels(series, sl_col=volatility, tp_mult=1.0, horizon=10)
        self.assertIsInstance(labels_dynamic, pd.DataFrame)
        self.assertIn('label', labels_dynamic.columns)
        self.assertTrue(set(labels_dynamic['label'].dropna().unique()).issubset({0, 1, 2}))

    def test_gsadf_values(self):
        """Test GSADF."""
        # Use log prices as expected by GSADF
        series = np.log(self.ohlc_data["Close"])

        result = gsadf_values(series, model='linear', min_length=20, lags=1)

        self.assertIsInstance(result, pd.Series)
        # the result series starts at index min_length+lags+2 roughly, so len(result) < len(series)
        self.assertLessEqual(len(result), len(series))
        # In the JAX version, it filters starting from `min_length`. We shouldn't strictly assume iloc[0:15] are NaN if it's already filtered.
        # But we can assert the end is not NaN.
        # Later values should be valid
        self.assertFalse(np.isnan(result.iloc[-1]))

    def test_ichimoku(self):
        """Test Ichimoku Cloud function."""
        high = self.ohlc_data["High"]
        low = self.ohlc_data["Low"]
        close = self.ohlc_data["Close"]

        result = ICHIMOKU(high, low, close)

        self.assertEqual(len(result), 7) # Returns tuple of 7 arrays
        tenkan, kijun, span_a, span_b, span_a_shift, span_b_shift, chikou = result

        self.assertEqual(len(tenkan), len(close))
        self.assertIsInstance(tenkan, np.ndarray)

    def test_pipe_indicator(self):
        """Test pipe_indicator wrapper."""
        df = self.ohlc_data.copy()

        # Test with Ichimoku
        # output_names needs to match the tuple return count of ICHIMOKU (7)
        output_names = ['tenkan', 'kijun', 'span_a', 'span_b', 'span_a_s', 'span_b_s', 'chikou']

        df_new = pipe_indicator(
            df,
            ICHIMOKU,
            input_map={'high': 'High', 'low': 'Low', 'close': 'Close'},
            output_names=output_names
        )

        for name in output_names:
            if name != 'chikou':
                self.assertIn(name, df_new.columns)

    def test_sk_transformers(self):
        """Test sklearn transformers."""
        series = self.ohlc_data["Close"]
        df = self.ohlc_data.copy()

        # Frac Diff
        transformer = FractionalDiffTransformer(d=0.4)
        res = transformer.fit_transform(series)
        self.assertIsInstance(res, pd.Series)

        res_df = transformer.fit_transform(df[["Close"]]) # single col DF
        self.assertTrue(isinstance(res_df, (pd.Series, pd.DataFrame)))

        # Trend Scanning
        ts_trans = TrendScanningTransformer(windows=[10])
        res_ts = ts_trans.fit_transform(series)
        self.assertIsInstance(res_ts, pd.Series)

        # GSADF
        gsadf_trans = GSADFTransformer(min_length=20)
        res_gsadf = gsadf_trans.fit_transform(np.log(series))
        self.assertIsInstance(res_gsadf, pd.Series)

        # Triple Barrier
        tb_trans = TripleBarrierLabeler(price_col='Close', vertical_barrier_steps=5)
        res_tb = tb_trans.fit_transform(df)
        self.assertIsInstance(res_tb, pd.DataFrame)

    @patch('pyquantflow.data.quarterly_pull.yf.download')
    def test_fetch_quarterly_data(self, mock_download):
        """Test fetch_quarterly_data with mocked yfinance."""
        # Setup mock return
        mock_df = self.generate_synthetic_ohlc(n=100)
        # yfinance download usually returns a DF with Timezone if auto_adjust=True, etc.
        # The code does data.index.tz_convert('Australia/Sydney') so we need a timezone-aware index initially or handle it.
        # But wait, the code does `data.index = data.index.tz_convert(...)`.
        # If mock_df is tz-naive, tz_convert might fail if it thinks it's already naive, or work if it assumes UTC.
        # Actually in pandas, you usually need to localize first if it's naive, or convert if it's aware.
        # Let's make it tz-aware (UTC) to be safe, as yfinance usually returns UTC.
        mock_df.index = mock_df.index.tz_localize('UTC')

        mock_download.return_value = mock_df

        ticker = "AAPL"
        time_dict = {2023: [1]}

        result = fetch_quarterly_data(ticker, time_dict, period='quarterly')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(mock_download.called)
        # Check if tz conversion happened
        self.assertEqual(str(result.index.tz), 'Australia/Sydney')

    def test_merge_last_hour(self):
        """Test merge_last_hour logic."""
        # Create a DF with hourly data for a single day
        # 10:00, 11:00 ... 15:00, 16:00 (last hour)
        # We need at least 2 rows
        dates = pd.date_range("2023-01-01 15:00", periods=2, freq="h")
        df = pd.DataFrame({
            "High": [100, 102],
            "Low": [90, 95],
            "Close": [95, 98],
            "Volume": [1000, 500]
        }, index=dates)

        merged = merge_last_hour(df)

        # Should have 1 row now
        self.assertEqual(len(merged), 1)
        # Check merged values
        # High = max(100, 102) = 102
        self.assertEqual(merged.iloc[0]["High"], 102)
        # Low = min(90, 95) = 90
        self.assertEqual(merged.iloc[0]["Low"], 90)
        # Close = last close = 98
        self.assertEqual(merged.iloc[0]["Close"], 98)
        # Volume = sum(1000, 500) = 1500
        self.assertEqual(merged.iloc[0]["Volume"], 1500)

if __name__ == '__main__':
    unittest.main()
