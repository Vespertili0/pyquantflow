import unittest
import os
import sqlite3
import json
import pandas as pd
from pyquantflow.backtesting.batchbacktest import BatchBacktester
from pyquantflow.strategies.example_strategy import SmaCross
from pyquantflow.data.database import DatabaseManager
from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.model.classifier import BaseQuantClassifier
import numpy as np

class MockClassifier(BaseQuantClassifier):
    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, sample_weight: np.ndarray | None = None) -> 'BaseQuantClassifier':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out['target_col'] = 1  # Dummy values
        return X_out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones((len(X), 2))

class TestBacktesting(unittest.TestCase):
    def setUp(self):
        # Database paths
        self.source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.results_db_path = os.path.join(os.path.dirname(__file__), "test_backtest_results.db")

        # Ensure source DB exists
        if not os.path.exists(self.source_db_path):
            self.fail(f"Source database file not found at {self.source_db_path}")

        # Initialize managers
        self.db_manager = DatabaseManager(db_path=self.source_db_path)
        self.backtester = BatchBacktester(results_db_path=self.results_db_path)

        self.tickers = ['FMG.AX', 'CBA.AX']

    def tearDown(self):
        # Clean up results DB
        if os.path.exists(self.results_db_path):
            os.remove(self.results_db_path)

    def test_run_batch_backtest_and_verify_results(self):
        # 1. Load data from stocks.db
        data_map = {}
        for ticker in self.tickers:
            df = self.db_manager.get_data(ticker)
            if not df.empty:
                data_map[ticker] = df
            else:
                print(f"Warning: No data found for {ticker} in stocks.db")

        self.assertTrue(data_map, "No data loaded from database for backtesting.")

        # 2. Run Backtest
        results = self.backtester.run_batch_backtest(
            SmaCross,
            data_map=data_map,
            cash=10000,
            commission=.002
        )

        # 3. Verify returned results structure
        self.assertIn('individual_results', results)
        self.assertIn('average_metrics', results)

        # Verify individual results
        for ticker in data_map.keys():
            self.assertIn(ticker, results['individual_results'])
            stats = results['individual_results'][ticker]

            # Check for some key metrics
            # Note: If backtest fails/errors, stats might contain 'Error' key
            if 'Error' in stats:
                self.fail(f"Backtest failed for {ticker}: {stats['Error']}")

            self.assertIn('Return [%]', stats)
            self.assertIn('Sharpe Ratio', stats)
            self.assertIn('# Trades', stats)
            self.assertIn('Max. Drawdown [%]', stats)

        # Verify averages (only if we have valid results)
        if results['average_metrics']:
            avg_metrics = results['average_metrics']
            self.assertIn('Return [%]', avg_metrics)
            self.assertIn('Sharpe Ratio', avg_metrics)

        # 4. Verify Database Storage (Should be empty before saving)
        conn = sqlite3.connect(self.results_db_path)
        cursor = conn.cursor()

        # Check that no results are stored yet
        cursor.execute("SELECT count(*) FROM backtest_results")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 0, "Database should be empty before explicit save")

        # 5. Save Results
        batch_name = self.backtester.save_batch_results()

        # Verify batch name format
        from datetime import datetime
        expected_prefix = datetime.now().strftime('%Y-%m-%d')
        self.assertTrue(batch_name.startswith(expected_prefix))
        self.assertTrue(batch_name.endswith(SmaCross.__name__))

        # 6. Verify Database Storage after saving
        for ticker in data_map.keys():
            cursor.execute("SELECT * FROM backtest_results WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            self.assertIsNotNone(row, f"No results found in database for {ticker}")

            # Verify batch name storage
            self.assertEqual(row[2], batch_name) # batch_run_name is the 3rd column (index 2)

            # Verify JSON content
            stored_metrics = json.loads(row[3]) # metrics is the 4th column (index 3)
            self.assertIn('Return [%]', stored_metrics)

        conn.close()

    def test_run_batch_backtest_with_asset_organiser(self):
        # 1. Load data from stocks.db
        data_map = {}
        for ticker in self.tickers:
            df = self.db_manager.get_data(ticker)
            if not df.empty:
                # Mock a target feature column
                df['target_col'] = 1
                data_map[ticker] = df
            else:
                print(f"Warning: No data found for {ticker} in stocks.db")

        self.assertTrue(data_map, "No data loaded from database for backtesting.")

        # 2. Setup AssetOrganiser
        mock_classifier = MockClassifier()
        # Ensure cutoff date is in the middle of our sample data
        cutoff_date = "2023-01-01"
        asset_organiser = AssetOrganiser(
            classifier=mock_classifier,
            data_map=data_map,
            cutoff_date=cutoff_date,
            target_features=['target_col']
        )
        asset_organiser.prepare_multi_asset_frame()
        asset_organiser.fit_classifier()

        # 3. Run Backtest
        results = self.backtester.run_batch_backtest(
            SmaCross,
            asset_organiser=asset_organiser,
            cash=10000,
            commission=.002
        )

        # 4. Verify returned results structure
        self.assertIn('individual_results', results)
        self.assertIn('average_metrics', results)

        # Verify individual results
        for ticker in data_map.keys():
            self.assertIn(ticker, results['individual_results'])
            stats = results['individual_results'][ticker]

            # Check for some key metrics
            if 'Error' in stats:
                self.fail(f"Backtest failed for {ticker}: {stats['Error']}")

            self.assertIn('Return [%]', stats)
            self.assertIn('Sharpe Ratio', stats)
            self.assertIn('# Trades', stats)
            self.assertIn('Max. Drawdown [%]', stats)

        # Verify averages (only if we have valid results)
        if results['average_metrics']:
            avg_metrics = results['average_metrics']
            self.assertIn('Return [%]', avg_metrics)
            self.assertIn('Sharpe Ratio', avg_metrics)


if __name__ == "__main__":
    unittest.main()
