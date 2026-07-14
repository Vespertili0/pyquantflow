import unittest
import json
from unittest.mock import patch

from pyquantflow.backtesting.backtest_database import BacktestDatabaseManager


class TestBacktestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db = BacktestDatabaseManager(":memory:")

    def tearDown(self):
        self.db.conn.close()

    def test_create_tables(self):
        # Check if the table 'backtest_results' was created
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_results'"
        )
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists)

        # Check table columns
        cursor.execute("PRAGMA table_info(backtest_results)")
        columns = {row[1] for row in cursor.fetchall()}
        self.assertTrue({"id", "ticker", "batch_run_name", "metrics"}.issubset(columns))

    def test_save_result(self):
        # Define test data
        ticker = "AAPL"
        batch_run_name = "test_run_01"
        result_dict = {"Return [%]": 15.5, "Sharpe Ratio": 1.2}

        # Save result
        self.db.save_result(ticker, result_dict, batch_run_name)

        # Query database to verify insertion
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT ticker, batch_run_name, metrics FROM backtest_results")
        rows = cursor.fetchall()

        self.assertEqual(len(rows), 1)
        db_ticker, db_batch, db_metrics = rows[0]

        self.assertEqual(db_ticker, ticker)
        self.assertEqual(db_batch, batch_run_name)

        # Verify JSON
        metrics_loaded = json.loads(db_metrics)
        self.assertEqual(metrics_loaded, result_dict)

    @patch("pyquantflow.backtesting.backtest_database.json.dumps")
    @patch("pyquantflow.backtesting.backtest_database.logger")
    def test_save_result_serialization_error(self, mock_logger, mock_dumps):
        # Mock json.dumps to raise an exception
        mock_dumps.side_effect = TypeError(
            "Object of type XXX is not JSON serializable"
        )

        ticker = "AAPL"
        batch_run_name = "test_run_01"
        result_dict = {
            "unserializable": set([1, 2, 3])
        }  # A set is normally not serializable

        # Call save_result
        self.db.save_result(ticker, result_dict, batch_run_name)

        # Verify error was logged
        mock_logger.error.assert_called_once()
        self.assertIn(
            f"Error serializing results for {ticker}:",
            mock_logger.error.call_args[0][0],
        )

        # Verify database is empty
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT count(*) FROM backtest_results")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
