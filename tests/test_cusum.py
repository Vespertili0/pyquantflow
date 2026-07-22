import unittest
import numpy as np
import pandas as pd
from pyquantflow.data.labels import get_cusum_events, calibrate_cusum_alpha
from pyquantflow.data.assetorganiser import AssetOrganiser


class TestCUSUMFilter(unittest.TestCase):
    def setUp(self):
        import os
        from pyquantflow.data.database import DatabaseManager

        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.prices = None

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                df = db_manager.get_data("CBA.AX")
                if not df.empty and len(df) >= 100:
                    self.prices = df["Close"].iloc[-100:].copy()
                db_manager.conn.close()
            except Exception:
                pass

        if self.prices is None:
            np.random.seed(42)
            # Create 100 days of price data
            self.dates = pd.date_range("2021-01-01", periods=100)
            # Price process: driftless random walk starting at 100
            self.prices = pd.Series(
                100.0 + np.cumsum(np.random.normal(0, 1.0, 100)), index=self.dates
            )
        self.dates = self.prices.index

    def test_cusum_constant_threshold(self):
        # Apply CUSUM with constant threshold on prepared returns
        returns = self.prices.pct_change()
        threshold = 0.02
        events = get_cusum_events(returns, threshold=threshold)

        self.assertIsInstance(events, pd.DatetimeIndex)
        # Since it is a random walk, we expect some events to trigger
        self.assertTrue(len(events) > 0)
        # Check that events are unique and sorted
        self.assertTrue(events.is_monotonic_increasing)
        self.assertTrue(events.is_unique)

        # Confirm all returned events exist in the original index
        for event in events:
            self.assertIn(event, returns.index)

    def test_cusum_dynamic_threshold(self):
        # Create a dynamic threshold Series (e.g. rolling volatility)
        returns = self.prices.pct_change()
        dynamic_threshold = returns.ewm(span=20).std() * 2.0

        events = get_cusum_events(returns, threshold=dynamic_threshold)
        self.assertIsInstance(events, pd.DatetimeIndex)
        self.assertTrue(len(events) > 0)

    def test_invalid_index_raises_type_error(self):
        # Test that non-DatetimeIndex raises a TypeError
        non_dt_prices = pd.Series([100.0, 101.0, 102.0], index=[0, 1, 2])
        with self.assertRaises(TypeError):
            get_cusum_events(non_dt_prices, threshold=0.01)

    def test_calibrate_cusum_alpha(self):
        # Target 5 events
        target = 5
        returns = self.prices.pct_change()
        alpha = calibrate_cusum_alpha(
            series=returns,
            target_events=target,
            alpha_min=0.5,
            alpha_max=6.0,
            alpha_step=0.1,
            span=20,
        )

        self.assertIsInstance(alpha, float)
        self.assertTrue(0.5 <= alpha <= 6.0)

        # Run CUSUM with the calibrated alpha to check if event count is near target
        vol = returns.ewm(span=20).std()
        threshold = alpha * vol
        events = get_cusum_events(returns, threshold)

        # Difference should be minimised
        self.assertTrue(abs(len(events) - target) <= 5)

    def test_calibrate_cusum_alpha_budget_mode(self):
        """Explicit budget objective must behave identically to legacy call."""
        target = 5
        returns = self.prices.pct_change()
        alpha = calibrate_cusum_alpha(
            series=returns,
            target_events=target,
            alpha_min=0.5,
            alpha_max=6.0,
            alpha_step=0.1,
            span=20,
            objective="budget",
        )

        self.assertIsInstance(alpha, float)
        self.assertTrue(0.5 <= alpha <= 6.0)

        # Confirm that the resulting event count is near the target
        vol = returns.ewm(span=20).std()
        threshold = alpha * vol
        events = get_cusum_events(returns, threshold)
        self.assertTrue(abs(len(events) - target) <= 5)

    def test_calibrate_cusum_alpha_uniqueness_mode(self):
        """Uniqueness objective must return a valid alpha within the search range."""
        returns = self.prices.pct_change().dropna()

        # Construct a synthetic t1 series: barrier ends 5 days after each bar
        t1 = pd.Series(
            returns.index + pd.Timedelta(days=5),
            index=returns.index,
            name="t1",
        )

        alpha = calibrate_cusum_alpha(
            series=returns,
            alpha_min=0.5,
            alpha_max=3.0,
            alpha_step=0.5,
            span=20,
            objective="uniqueness",
            t1=t1,
        )

        self.assertIsInstance(alpha, float)
        self.assertTrue(0.5 <= alpha <= 3.0)

    def test_calibrate_cusum_alpha_uniqueness_no_t1_raises(self):
        """Passing objective='uniqueness' without t1 must raise ValueError."""
        returns = self.prices.pct_change()
        with self.assertRaises(ValueError):
            calibrate_cusum_alpha(
                series=returns,
                target_events=5,
                objective="uniqueness",
                t1=None,
            )

    def test_calibrate_cusum_alpha_budget_no_target_raises(self):
        """Passing objective='budget' without target_events must raise ValueError."""
        returns = self.prices.pct_change()
        with self.assertRaises(ValueError):
            calibrate_cusum_alpha(
                series=returns,
                target_events=None,
                objective="budget",
            )


class TestAssetOrganiserDownsampling(unittest.TestCase):
    def setUp(self):
        import os
        from pyquantflow.data.database import DatabaseManager

        self.data_map = {}
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                for ticker in ["FMG.AX", "CBA.AX"]:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 100:
                        df = df.iloc[-100:].copy()
                        np.random.seed(42)
                        df["close"] = df["Close"]
                        df["feature1"] = np.random.randn(len(df))
                        df["target"] = np.random.randint(0, 2, len(df))
                        self.data_map[ticker] = df
                db_manager.conn.close()
            except Exception:
                pass

        if len(self.data_map) < 2:
            np.random.seed(42)
            dates = pd.date_range("2020-01-01", periods=100)

            prices_a = 100.0 + np.cumsum(np.random.normal(0, 1.0, 100))
            df_a = pd.DataFrame(
                {
                    "close": prices_a,
                    "feature1": np.random.randn(100),
                    "target": np.random.randint(0, 2, 100),
                },
                index=dates,
            )
            df_a.index.name = "datetime"

            prices_b = 100.0 + np.cumsum(np.random.normal(0, 1.0, 100))
            df_b = pd.DataFrame(
                {
                    "close": prices_b,
                    "feature1": np.random.randn(100),
                    "target": np.random.randint(0, 2, 100),
                },
                index=dates,
            )
            df_b.index.name = "datetime"

            self.data_map = {"AAA": df_a, "BBB": df_b}

        self.ticker_1 = list(self.data_map.keys())[0]
        self.ticker_2 = list(self.data_map.keys())[1]
        self.cutoff_date = self.data_map[self.ticker_1].index[60]
        self.global_event_1 = self.data_map[self.ticker_1].index[2]
        self.global_event_2 = self.data_map[self.ticker_1].index[4]
        self.ticker_event_1 = self.data_map[self.ticker_1].index[1]
        self.ticker_event_2 = self.data_map[self.ticker_2].index[4]

    def test_downsample_global_events(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # Check initial length (at least ~200 rows)
        self.assertTrue(len(organiser.multi_asset) >= 200)

        # Apply global filter for two specific dates
        global_events = [
            pd.Timestamp(self.global_event_1),
            pd.Timestamp(self.global_event_2),
        ]
        organiser.downsample_to_events(global_events)

        # Length should now be 2 dates x 2 tickers = 4 rows
        self.assertEqual(len(organiser.multi_asset), 4)

        # Confirm all datetimes are within the filter set
        dts = organiser.multi_asset.index.get_level_values("datetime")
        for dt in dts:
            self.assertIn(pd.Timestamp(dt), global_events)

    def test_downsample_ticker_specific_events(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        events_dict = {
            self.ticker_1: pd.DatetimeIndex([self.ticker_event_1]),
            self.ticker_2: pd.DatetimeIndex([self.ticker_event_2]),
        }

        organiser.downsample_to_events(events_dict)

        # Should only have two rows left
        self.assertEqual(len(organiser.multi_asset), 2)

        # Check indexes specifically
        idx = organiser.multi_asset.index
        self.assertIn((pd.Timestamp(self.ticker_event_1), self.ticker_1), idx)
        self.assertIn((pd.Timestamp(self.ticker_event_2), self.ticker_2), idx)

    def test_downsample_to_cusum_events(self):
        # Pre-calculate returns column beforehand
        for tk, df in self.data_map.items():
            df["returns"] = df["close"].pct_change()

        # Setup organiser with cutoff
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # 60 train days, 40 test days. Target 5 events on train set.
        calibrated_alphas = organiser.downsample_to_cusum_events(
            target_events_train=5,
            filter_col="returns",
            span=20,
        )

        self.assertIsInstance(calibrated_alphas, dict)
        self.assertIn(self.ticker_1, calibrated_alphas)
        self.assertIn(self.ticker_2, calibrated_alphas)

        # Confirm alphas are calibrated floats
        self.assertIsInstance(calibrated_alphas[self.ticker_1], float)
        self.assertIsInstance(calibrated_alphas[self.ticker_2], float)

        # Confirm dataset was successfully downsampled
        self.assertTrue(len(organiser.multi_asset) < 200)

        # Confirm splits exist
        self.assertIsNotNone(organiser.multi_asset_train)
        self.assertIsNotNone(organiser.multi_asset_test)

    def test_downsample_to_cusum_events_external_vol(self):
        # Pre-calculate returns and simulated volatility beforehand
        for tk, df in self.data_map.items():
            df["returns"] = df["close"].pct_change()
            df["rs_vol"] = df["returns"].ewm(span=20).std()

        # Setup organiser with cutoff
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # Call with both filter_col and vol_col
        calibrated_alphas = organiser.downsample_to_cusum_events(
            target_events_train=5,
            filter_col="returns",
            vol_col="rs_vol",
            span=20,
        )

        self.assertIsInstance(calibrated_alphas, dict)
        self.assertIn(self.ticker_1, calibrated_alphas)
        self.assertIn(self.ticker_2, calibrated_alphas)
        self.assertIsInstance(calibrated_alphas[self.ticker_1], float)
        self.assertTrue(len(organiser.multi_asset) < 200)

    def test_invalid_type_raises_error(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date=self.cutoff_date,
            target_features=["target"],
        )
        with self.assertRaises(TypeError):
            organiser.downsample_to_events(12345)


if __name__ == "__main__":
    unittest.main()
