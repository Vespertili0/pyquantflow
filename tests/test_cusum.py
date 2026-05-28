import unittest
import numpy as np
import pandas as pd
from pyquantflow.data.labels import get_cusum_events, calibrate_cusum_alpha
from pyquantflow.data.assetorganiser import AssetOrganiser


class TestCUSUMFilter(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create 100 days of price data
        self.dates = pd.date_range("2021-01-01", periods=100)
        # Price process: driftless random walk starting at 100
        self.prices = pd.Series(
            100.0 + np.cumsum(np.random.normal(0, 1.0, 100)), index=self.dates
        )

    def test_cusum_constant_threshold(self):
        # Apply CUSUM with constant threshold
        threshold = 0.02
        events = get_cusum_events(self.prices, threshold=threshold)

        self.assertIsInstance(events, pd.DatetimeIndex)
        # Since it is a random walk, we expect some events to trigger
        self.assertTrue(len(events) > 0)
        # Check that events are unique and sorted
        self.assertTrue(events.is_monotonic_increasing)
        self.assertTrue(events.is_unique)

        # Confirm all returned events exist in the original index
        for event in events:
            self.assertIn(event, self.prices.index)

    def test_cusum_dynamic_threshold(self):
        # Create a dynamic threshold Series (e.g. rolling volatility)
        returns = self.prices.pct_change()
        dynamic_threshold = returns.ewm(span=20).std() * 2.0

        events = get_cusum_events(self.prices, threshold=dynamic_threshold)
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
        alpha = calibrate_cusum_alpha(
            prices=self.prices,
            target_events=target,
            alpha_min=0.5,
            alpha_max=3.0,
            alpha_step=0.1,
            span=20,
        )

        self.assertIsInstance(alpha, float)
        self.assertTrue(0.5 <= alpha <= 3.0)

        # Run CUSUM with the calibrated alpha to check if event count is near target
        returns = self.prices.pct_change()
        vol = returns.ewm(span=20).std()
        threshold = alpha * vol
        events = get_cusum_events(self.prices, threshold)

        # Difference should be minimized
        self.assertTrue(abs(len(events) - target) <= 5)


class TestAssetOrganiserDownsampling(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100)

        # Price process starts at 100.0
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

    def test_downsample_global_events(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date="2020-03-01",
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # Check initial length (100 dates x 2 tickers = 200 rows)
        self.assertEqual(len(organiser.multi_asset), 200)

        # Apply global filter for two specific dates
        global_events = [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05")]
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
            cutoff_date="2020-03-01",
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # AAA triggers on 2020-01-02, BBB triggers on 2020-01-05
        events_dict = {
            "AAA": pd.DatetimeIndex(["2020-01-02"]),
            "BBB": pd.DatetimeIndex(["2020-01-05"]),
        }

        organiser.downsample_to_events(events_dict)

        # Should only have two rows left (one for AAA, one for BBB)
        self.assertEqual(len(organiser.multi_asset), 2)

        # Check indexes specifically
        idx = organiser.multi_asset.index
        self.assertIn((pd.Timestamp("2020-01-02"), "AAA"), idx)
        self.assertIn((pd.Timestamp("2020-01-05"), "BBB"), idx)

    def test_downsample_to_cusum_events(self):
        # Setup organiser with cutoff
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date="2020-03-01",
            target_features=["target"],
        )
        organiser.prepare_multi_asset_frame()

        # 60 train days, 40 test days. Target 5 events on train set.
        calibrated_alphas = organiser.downsample_to_cusum_events(
            target_events_train=5,
            price_col="close",
            span=20,
        )

        self.assertIsInstance(calibrated_alphas, dict)
        self.assertIn("AAA", calibrated_alphas)
        self.assertIn("BBB", calibrated_alphas)

        # Confirm alphas are calibrated floats
        self.assertIsInstance(calibrated_alphas["AAA"], float)
        self.assertIsInstance(calibrated_alphas["BBB"], float)

        # Confirm dataset was successfully downsampled
        self.assertTrue(len(organiser.multi_asset) < 200)

        # Confirm splits exist
        self.assertIsNotNone(organiser.multi_asset_train)
        self.assertIsNotNone(organiser.multi_asset_test)

    def test_invalid_type_raises_error(self):
        organiser = AssetOrganiser(
            data_map=self.data_map,
            cutoff_date="2020-03-01",
            target_features=["target"],
        )
        with self.assertRaises(TypeError):
            organiser.downsample_to_events(12345)


if __name__ == "__main__":
    unittest.main()
