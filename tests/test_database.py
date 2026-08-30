import unittest
from unittest.mock import patch
import pandas as pd
from pyquantflow.data.database import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use in-memory database for testing
        self.db = DatabaseManager(":memory:")

    def tearDown(self):
        self.db.conn.close()

    @patch("pyquantflow.data.database.fetch_quarterly_data")
    def test_add_ticker(self, mock_fetch):
        # Setup mock return data
        dates = pd.date_range(
            start="2023-01-01", periods=5, freq="D", tz="Australia/Sydney"
        )
        data = {
            "Open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "High": [15.0, 16.0, 17.0, 18.0, 19.0],
            "Low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "Close": [12.0, 13.0, 14.0, 15.0, 16.0],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }
        mock_df = pd.DataFrame(data, index=dates)
        mock_df.index.name = "Datetime"
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
        cursor.execute(
            "SELECT count(*) FROM price_data WHERE ticker_id = ?", (ticker_id,)
        )
        count = cursor.fetchone()[0]
        self.assertEqual(count, 5)

        # Verify get_data
        df = self.db.get_data(ticker)
        self.assertEqual(len(df), 5)
        self.assertEqual(df.iloc[0]["Open"], 10.0)

    @patch("pyquantflow.data.database.yf.download")
    @patch("pyquantflow.data.database.fetch_quarterly_data")
    def test_update_ticker(self, mock_fetch, mock_yf_download):
        # 1. Add ticker first
        dates_initial = pd.date_range(
            start="2023-01-01", periods=2, freq="D", tz="Australia/Sydney"
        )
        data_initial = {
            "Open": [10.0, 11.0],
            "High": [15.0, 16.0],
            "Low": [9.0, 10.0],
            "Close": [12.0, 13.0],
            "Volume": [1000, 1100],
        }
        mock_df_initial = pd.DataFrame(data_initial, index=dates_initial)
        mock_fetch.return_value = mock_df_initial

        ticker = "TEST.AX"
        self.db.add_ticker(ticker)

        # 2. Update ticker
        # Ensure dates_update is AFTER dates_initial
        dates_update = pd.date_range(
            start="2023-01-03", periods=2, freq="D", tz="Australia/Sydney"
        )
        data_update = {
            "Open": [20.0, 21.0],
            "High": [25.0, 26.0],
            "Low": [19.0, 20.0],
            "Close": [22.0, 23.0],
            "Volume": [2000, 2100],
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

    @patch.object(DatabaseManager, "_update_ticker_internal")
    @patch.object(DatabaseManager, "add_ticker")
    def test_update_tickers_batch(self, mock_add_ticker, mock_update_ticker_internal):
        tickers = ["TEST1.AX", "TEST2.AX", "TEST3.AX"]

        # Insert some dummy records so they're found in ticker_info dict
        cursor = self.db.conn.cursor()
        for i, t in enumerate(tickers[:2]):
            cursor.execute(self.db._SQL_INSERT_TICKER, (t, "1h", "2023-01-01"))
            ticker_id = cursor.lastrowid
            self.db._insert_price_data(
                ticker_id,
                pd.DataFrame(
                    {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
                    index=pd.date_range("2023-01-01", periods=1, tz="UTC"),
                ),
            )
        self.db.conn.commit()

        # Test that update_tickers_batch calls internal functions correctly
        self.db.update_tickers_batch(tickers)

        # Assert that _update_ticker_internal was called twice (for TEST1.AX and TEST2.AX)
        self.assertEqual(mock_update_ticker_internal.call_count, 2)

        # Assert that add_ticker was called once (for TEST3.AX)
        mock_add_ticker.assert_called_once_with(
            "TEST3.AX", commit=False, skip_check=True
        )

    @patch.object(DatabaseManager, "_update_ticker_internal")
    @patch.object(DatabaseManager, "add_ticker")
    def test_update_tickers_batch_handles_exceptions(
        self, mock_add_ticker, mock_update_ticker_internal
    ):
        tickers = ["TEST1.AX", "ERROR.AX", "TEST3.AX"]

        # Insert some dummy records so they're found in ticker_info dict
        cursor = self.db.conn.cursor()
        for i, t in enumerate(tickers):
            cursor.execute(self.db._SQL_INSERT_TICKER, (t, "1h", "2023-01-01"))
        self.db.conn.commit()

        # Make it throw an exception for ERROR.AX but not for others
        def side_effect(
            ticker,
            ticker_id,
            interval,
            commit=False,
            last_date_str=None,
            skip_max_date_lookup=False,
        ):
            if ticker == "ERROR.AX":
                raise Exception("Mock error")

        mock_update_ticker_internal.side_effect = side_effect

        # Should not raise an exception overall
        self.db.update_tickers_batch(tickers)

        # Should still process all tickers
        self.assertEqual(mock_update_ticker_internal.call_count, len(tickers))

    def test_insert_price_data_utc_normalisation(self):
        """
        Req 1.2: Datetime strings written to price_data must carry the UTC offset (+00:00),
        regardless of the timezone of the DataFrame passed to _insert_price_data.
        """
        # Build a Sydney-localised DataFrame (UTC+10/+11) — what fetch_quarterly_data used to return
        dates_sydney = pd.date_range(
            start="2023-01-04", periods=3, freq="D", tz="Australia/Sydney"
        )
        mock_df = pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0],
                "High": [1.5, 2.5, 3.5],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.2, 2.2, 3.2],
                "Volume": [100, 200, 300],
            },
            index=dates_sydney,
        )
        mock_df.index.name = "Datetime"

        # Insert directly via the private method (bypass yfinance)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "INSERT INTO tickers (ticker, interval, last_updated) VALUES (?, ?, ?)",
            ("UTC_TEST.AX", "1d", "2023-01-01"),
        )
        ticker_id = cursor.lastrowid
        self.db._insert_price_data(ticker_id, mock_df)
        self.db.conn.commit()

        # Read back raw datetime strings from SQLite
        cursor.execute(
            "SELECT datetime FROM price_data WHERE ticker_id = ?", (ticker_id,)
        )
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 3)

        for (dt_str,) in rows:
            # UTC offset is represented as +00:00 in the ISO-format string
            self.assertIn(
                "+00:00",
                dt_str,
                msg=f"Expected UTC offset in '{dt_str}' but found a different offset.",
            )

    def test_create_tables_migration(self):
        """Test that the interval column is added if it does not exist."""
        # Create table without interval
        self.db.conn.execute("DROP TABLE IF EXISTS tickers")
        self.db.conn.execute(
            """
            CREATE TABLE tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                first_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP
            )
            """
        )
        self.db.conn.commit()
        
        # Call create_tables to trigger migration
        self.db.create_tables()
        
        cursor = self.db.conn.cursor()
        cursor.execute("PRAGMA table_info(tickers)")
        columns = [info[1] for info in cursor.fetchall()]
        self.assertIn("interval", columns)

    @patch("pyquantflow.data.database.fetch_quarterly_data")
    def test_add_ticker_empty_df(self, mock_fetch):
        """Test add_ticker when fetched data is empty."""
        mock_fetch.return_value = pd.DataFrame()
        self.db.add_ticker("EMPTY.AX")
        
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM tickers WHERE ticker = ?", ("EMPTY.AX",))
        self.assertIsNone(cursor.fetchone())

    @patch.object(DatabaseManager, "add_ticker")
    def test_update_ticker_not_found(self, mock_add):
        """Test update_ticker falls back to add_ticker if not found."""
        self.db.update_ticker("NEW.AX")
        mock_add.assert_called_once_with("NEW.AX", commit=True)

    def test_insert_price_data_empty(self):
        """Test _insert_price_data with empty DataFrame."""
        # Should not raise any error
        self.db._insert_price_data(1, pd.DataFrame())

    def test_get_data_not_found(self):
        """Test get_data returns empty DataFrame for unknown ticker."""
        df = self.db.get_data("UNKNOWN.AX")
        self.assertTrue(df.empty)

if __name__ == "__main__":
    unittest.main()

