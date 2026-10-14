import unittest
from unittest.mock import patch

import pandas as pd

from pyquantflow.data.quarterly_pull import fetch_quarterly_data


class TestQuarterlyPull(unittest.TestCase):
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
        self.assertIn("Failed to fetch data for 2023 Q1: Exception", cm.output[0])

    @patch("pyquantflow.data.quarterly_pull.yf.download")
    def test_fetch_quarterly_data_continues_after_exception(self, mock_download):
        """
        When Q1 raises an exception, Q2 must still be fetched.
        The returned DataFrame must contain the Q2 data only.
        """
        dates_q2 = pd.date_range(
            "2023-04-01", periods=2, freq="D", tz="America/New_York"
        )
        df_q2 = pd.DataFrame({"Close": [20.0, 21.0]}, index=dates_q2)

        # Q1 raises; Q2 succeeds
        mock_download.side_effect = [Exception("transient error"), df_q2]

        time_dict = {"2023": [1, 2]}

        with self.assertLogs("pyquantflow.data.quarterly_pull", level="ERROR") as cm:
            result = fetch_quarterly_data("AAPL", time_dict)

        self.assertFalse(result.empty)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["Close"].values), [20.0, 21.0])
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertIn("Failed to fetch data for 2023 Q1: Exception", cm.output[0])
        self.assertEqual(mock_download.call_count, 2)


if __name__ == "__main__":
    unittest.main()
