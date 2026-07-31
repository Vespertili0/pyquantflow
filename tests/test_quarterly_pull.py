import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from pyquantflow.data.quarterly_pull import merge_last_hour, fetch_quarterly_data


class TestQuarterlyPull(unittest.TestCase):
    def test_merge_last_hour_basic(self):
        """Test merging two hours on the same day."""
        dates = pd.date_range("2023-01-01 10:00:00", periods=2, freq="h")
        df = pd.DataFrame(
            {
                "High": [10.0, 12.0],
                "Low": [8.0, 9.0],
                "Close": [11.0, 11.5],
                "Volume": [100, 200],
            },
            index=dates,
        )

        merged = merge_last_hour(df)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.index[0], pd.Timestamp("2023-01-01 10:00:00"))
        self.assertEqual(merged.at[merged.index[0], "High"], 12.0)  # max(10, 12)
        self.assertEqual(merged.at[merged.index[0], "Low"], 8.0)    # min(8, 9)
        self.assertEqual(merged.at[merged.index[0], "Close"], 11.5) # last close
        self.assertEqual(merged.at[merged.index[0], "Volume"], 300) # sum(100, 200)

    def test_merge_last_hour_multiple_days(self):
        """Test merging across multiple days."""
        dates = pd.DatetimeIndex([
            "2023-01-01 10:00:00", "2023-01-01 11:00:00",
            "2023-01-02 10:00:00", "2023-01-02 11:00:00", "2023-01-02 12:00:00"
        ])
        df = pd.DataFrame(
            {
                "High": [10.0, 12.0, 15.0, 16.0, 14.0],
                "Low": [8.0, 9.0, 13.0, 12.0, 11.0],
                "Close": [11.0, 11.5, 14.5, 13.5, 12.5],
                "Volume": [100, 200, 300, 400, 500],
            },
            index=dates,
        )

        merged = merge_last_hour(df)

        self.assertEqual(len(merged), 3) # 1 from day 1, 2 from day 2
        
        # Check day 1
        day1_idx = pd.Timestamp("2023-01-01 10:00:00")
        self.assertEqual(merged.at[day1_idx, "High"], 12.0)
        self.assertEqual(merged.at[day1_idx, "Low"], 8.0)
        self.assertEqual(merged.at[day1_idx, "Close"], 11.5)
        self.assertEqual(merged.at[day1_idx, "Volume"], 300)

        # Check day 2
        day2_idx1 = pd.Timestamp("2023-01-02 10:00:00")
        day2_idx2 = pd.Timestamp("2023-01-02 11:00:00") # 12:00 is merged into 11:00
        
        self.assertEqual(merged.at[day2_idx1, "High"], 15.0)
        self.assertEqual(merged.at[day2_idx1, "Low"], 13.0)
        self.assertEqual(merged.at[day2_idx1, "Close"], 14.5)
        self.assertEqual(merged.at[day2_idx1, "Volume"], 300)

        self.assertEqual(merged.at[day2_idx2, "High"], 16.0) # max(16, 14)
        self.assertEqual(merged.at[day2_idx2, "Low"], 11.0)  # min(12, 11)
        self.assertEqual(merged.at[day2_idx2, "Close"], 12.5) # last close
        self.assertEqual(merged.at[day2_idx2, "Volume"], 900) # sum(400, 500)


    def test_merge_last_hour_single_hour_day(self):
        """Test behavior when a day has only one hour."""
        dates = pd.date_range("2023-01-01 10:00:00", periods=1, freq="h")
        df = pd.DataFrame(
            {
                "High": [10.0],
                "Low": [8.0],
                "Close": [11.0],
                "Volume": [100],
            },
            index=dates,
        )

        merged = merge_last_hour(df)

        # Should be unchanged
        self.assertEqual(len(merged), 1)
        pd.testing.assert_frame_equal(df, merged)

    @patch("pyquantflow.data.quarterly_pull.yf.download")
    def test_fetch_quarterly_data_success(self, mock_download):
        """Test successful fetch and concatenation of quarterly data."""
        
        # Setup mock return data
        dates1 = pd.date_range("2023-01-01", periods=2, freq="D", tz="America/New_York")
        df1 = pd.DataFrame({"Close": [10, 11]}, index=dates1)
        
        dates2 = pd.date_range("2023-04-01", periods=2, freq="D", tz="America/New_York")
        df2 = pd.DataFrame({"Close": [12, 13]}, index=dates2)

        # Mock download to return df1 for Q1 and df2 for Q2
        mock_download.side_effect = [df1, df2]

        time_dict = {"2023": [1, 2]}
        result = fetch_quarterly_data("AAPL", time_dict)

        self.assertEqual(len(result), 4)
        self.assertEqual(mock_download.call_count, 2)
        
        # Verify UTC conversion
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertEqual(list(result["Close"].values), [10, 11, 12, 13])

    def test_fetch_quarterly_data_invalid_period(self):
        """Test fetch with invalid period."""
        time_dict = {"2023": [1]}
        with self.assertRaises(ValueError):
            fetch_quarterly_data("AAPL", time_dict, period="yearly")

    @patch("pyquantflow.data.quarterly_pull.yf.download")
    def test_fetch_quarterly_data_exception_handling(self, mock_download):
        """Test handling of exceptions during fetch."""
        
        # Mock download to raise an exception
        mock_download.side_effect = Exception("API Error")

        time_dict = {"2023": [1]}
        
        # Test that exception is caught and logged, returning empty DataFrame
        with self.assertLogs("pyquantflow.data.quarterly_pull", level="ERROR") as cm:
            result = fetch_quarterly_data("AAPL", time_dict)
            
        self.assertTrue(result.empty)
        self.assertIn("Failed to fetch data for 2023 Q1: API Error", cm.output[0])

if __name__ == "__main__":
    unittest.main()
