import unittest
import pandas as pd

from pyquantflow.data.utils import (
    align_and_ffill_multiasset,
    generate_ema_ribbon_names,
    restructure_map_2_multiasset_df,
)


class TestUtils(unittest.TestCase):
    def test_align_and_ffill_multiasset_basic(self):
        # Create a sample multiasset dataframe with missing values
        dates = pd.date_range("2023-01-01", periods=3)

        # Missing AAPL on 2023-01-02, missing MSFT on 2023-01-03
        data = {
            "datetime": [dates[0], dates[0], dates[1], dates[2]],
            "ticker": ["AAPL", "MSFT", "MSFT", "AAPL"],
            "price": [150.0, 250.0, 255.0, 155.0],
        }
        df = pd.DataFrame(data).set_index(["datetime", "ticker"])

        aligned_df = align_and_ffill_multiasset(df)

        # Expected:
        # AAPL at 01-02 is ffilled from 01-01 -> 150.0
        # MSFT at 01-03 is ffilled from 01-02 -> 255.0

        self.assertEqual(len(aligned_df), 6)  # 3 dates * 2 tickers

        aapl_0102 = aligned_df.loc[(dates[1], "AAPL"), "price"]
        self.assertEqual(aapl_0102, 150.0)

        msft_0103 = aligned_df.loc[(dates[2], "MSFT"), "price"]
        self.assertEqual(msft_0103, 255.0)

    def test_align_and_ffill_multiasset_drop_leading_nan(self):
        # If a ticker is missing data for the first timestamps, ffill won't work,
        # so those rows will be NaN. The function uses dropna(), so those rows should be dropped.

        dates = pd.date_range("2023-01-01", periods=3)

        data = {
            "datetime": [dates[0], dates[1], dates[1], dates[2], dates[2]],
            "ticker": ["AAPL", "AAPL", "MSFT", "AAPL", "MSFT"],
            "price": [150.0, 152.0, 250.0, 155.0, 255.0],
        }
        # MSFT is missing on 2023-01-01

        df = pd.DataFrame(data).set_index(["datetime", "ticker"])

        aligned_df = align_and_ffill_multiasset(df)

        # AAPL will have 3 rows
        # MSFT will have 2 rows (because 2023-01-01 is NaN and dropped)
        self.assertEqual(len(aligned_df), 5)

        self.assertNotIn((dates[0], "MSFT"), aligned_df.index)

    def test_align_and_ffill_multiasset_full_grid(self):
        # Ensure it works smoothly with a complete grid without modifications
        dates = pd.date_range("2023-01-01", periods=2)
        tickers = ["AAPL", "MSFT"]

        full_index = pd.MultiIndex.from_product(
            [dates, tickers], names=["datetime", "ticker"]
        )
        df = pd.DataFrame({"price": [100, 200, 110, 210]}, index=full_index)

        aligned_df = align_and_ffill_multiasset(df)

        # DataFrame should be unchanged
        pd.testing.assert_frame_equal(df, aligned_df)

    def test_generate_ema_ribbon_names_default(self):
        names = generate_ema_ribbon_names()
        self.assertEqual(len(names), 24)
        self.assertEqual(names[0], "EMAR_RS_VOL")
        self.assertEqual(names[-1], "EMAR_VELOCITY_9v10")
        self.assertIn("EMAR_MACRO", names)
        self.assertIn("EMAR_STD", names)

    def test_generate_ema_ribbon_names_custom(self):
        names = generate_ema_ribbon_names(prefix="CUSTOM", M=3)
        self.assertEqual(len(names), 10)
        expected = [
            "CUSTOM_RS_VOL",
            "CUSTOM_MICRO_1v2",
            "CUSTOM_MICRO_2v3",
            "CUSTOM_MACRO",
            "CUSTOM_STD",
            "CUSTOM_SKEW",
            "CUSTOM_KURT",
            "CUSTOM_CONSENSUS_RANK",
            "CUSTOM_VELOCITY_1v2",
            "CUSTOM_VELOCITY_2v3",
        ]
        self.assertEqual(names, expected)

    def test_restructure_map_empty(self):
        # Empty dict should return empty DataFrame
        result = restructure_map_2_multiasset_df({})
        self.assertTrue(result.empty)
        self.assertIsInstance(result, pd.DataFrame)

    def test_restructure_map_basic(self):
        # Test basic functionality with no timezone
        dates = pd.date_range("2023-01-01", periods=2)
        df1 = pd.DataFrame({"price": [100.0, 105.0]}, index=dates)
        df1.index.name = "datetime"
        df2 = pd.DataFrame({"price": [200.0, 210.0]}, index=dates)
        df2.index.name = "datetime"

        df_dict = {"AAPL": df1, "MSFT": df2}
        result = restructure_map_2_multiasset_df(df_dict)

        self.assertEqual(result.index.names, ["datetime", "ticker"])
        self.assertEqual(len(result), 4)

        # Check values and sorting
        aapl_0101 = result.loc[(dates[0], "AAPL"), "price"]
        self.assertEqual(aapl_0101, 100.0)
        msft_0102 = result.loc[(dates[1], "MSFT"), "price"]
        self.assertEqual(msft_0102, 210.0)

    def test_restructure_map_with_timezone(self):
        # Test that timezone is preserved
        dates = pd.date_range("2023-01-01", periods=2, tz="UTC")
        df1 = pd.DataFrame({"price": [100.0, 105.0]}, index=dates)
        df1.index.name = "datetime"

        df_dict = {"AAPL": df1}
        result = restructure_map_2_multiasset_df(df_dict)

        self.assertEqual(str(result.index.get_level_values("datetime").tz), "UTC")

    def test_restructure_map_no_index_name(self):
        # Test when input df has no index name, it should be named 'datetime'
        dates = pd.date_range("2023-01-01", periods=2)
        df1 = pd.DataFrame({"price": [100.0, 105.0]}, index=dates)
        df1.index.name = None  # Ensure it is None

        df_dict = {"AAPL": df1}
        result = restructure_map_2_multiasset_df(df_dict)

        self.assertEqual(result.index.names, ["datetime", "ticker"])


if __name__ == "__main__":
    unittest.main()
