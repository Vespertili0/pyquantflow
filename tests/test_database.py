import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlite3
from pyquantflow.data.database import DatabaseManager
from datetime import datetime

class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        # Use in-memory database for testing
        self.db = DatabaseManager(":memory:")

    def tearDown(self):
        self.db.conn.close()

    @patch('pyquantflow.data.database.fetch_quarterly_data')
    def test_add_ticker(self, mock_fetch):
        # Setup mock return data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D', tz='Australia/Sydney')
        data = {
            'Open': [10.0, 11.0, 12.0, 13.0, 14.0],
            'High': [15.0, 16.0, 17.0, 18.0, 19.0],
            'Low': [9.0, 10.0, 11.0, 12.0, 13.0],
            'Close': [12.0, 13.0, 14.0, 15.0, 16.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        mock_df = pd.DataFrame(data, index=dates)
        mock_df.index.name = 'Datetime'
        mock_fetch.return_value = mock_df

        ticker = "TEST.AX"
        self.db.add_ticker(ticker, start_year=2023)

        # Verify ticker was added
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, ticker FROM tickers WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        ticker_id = row[0]
        self.assertEqual(row[1], ticker)

        # Verify data was inserted
        cursor.execute("SELECT count(*) FROM price_data WHERE ticker_id = ?", (ticker_id,))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 5)

        # Verify get_data
        df = self.db.get_data(ticker)
        self.assertEqual(len(df), 5)
        self.assertEqual(df.iloc[0]['Open'], 10.0)

    @patch('pyquantflow.data.database.yf.download')
    @patch('pyquantflow.data.database.fetch_quarterly_data')
    def test_update_ticker(self, mock_fetch, mock_yf_download):
        # 1. Add ticker first
        dates_initial = pd.date_range(start='2023-01-01', periods=2, freq='D', tz='Australia/Sydney')
        data_initial = {
            'Open': [10.0, 11.0],
            'High': [15.0, 16.0],
            'Low': [9.0, 10.0],
            'Close': [12.0, 13.0],
            'Volume': [1000, 1100]
        }
        mock_df_initial = pd.DataFrame(data_initial, index=dates_initial)
        mock_fetch.return_value = mock_df_initial

        ticker = "TEST.AX"
        self.db.add_ticker(ticker)

        # 2. Update ticker
        # Ensure dates_update is AFTER dates_initial
        dates_update = pd.date_range(start='2023-01-03', periods=2, freq='D', tz='Australia/Sydney')
        data_update = {
            'Open': [20.0, 21.0],
            'High': [25.0, 26.0],
            'Low': [19.0, 20.0],
            'Close': [22.0, 23.0],
            'Volume': [2000, 2100]
        }
        mock_df_update = pd.DataFrame(data_update, index=dates_update)
        mock_yf_download.return_value = mock_df_update

        self.db.update_ticker(ticker)

        # Verify total data
        df = self.db.get_data(ticker)
        self.assertEqual(len(df), 4)

        # Verify timestamps are correct in DB
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT last_updated FROM tickers WHERE ticker = ?", (ticker,))
        last_updated = cursor.fetchone()[0]
        self.assertIsNotNone(last_updated)

if __name__ == '__main__':
    unittest.main()
