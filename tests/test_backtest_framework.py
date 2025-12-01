import unittest
import os
import pandas as pd
from stock_package.backtest_framework import BatchBacktester
from stock_package.strategies.example_strategy import SmaCross

class TestBacktestFramework(unittest.TestCase):
    def setUp(self):
        # Locate the database file relative to this test file
        self.db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        if not os.path.exists(self.db_path):
            self.fail(f"Database file not found at {self.db_path}")
        
        self.tickers = ['FMG.AX', 'CBA.AX']
        self.backtester = BatchBacktester(db_path=self.db_path)

    def test_run_batch_backtest(self):
        # Run backtest
        results = self.backtester.run_batch_backtest(
            self.tickers,
            SmaCross,
            cash=10000,
            commission=.002
        )
        
        # Verify structure
        self.assertIn('individual_results', results)
        self.assertIn('average_metrics', results)

        # Verify individual results
        for ticker in self.tickers:
            self.assertIn(ticker, results['individual_results'])
            stats = results['individual_results'][ticker]

            # Check for some key metrics
            self.assertIn('Return [%]', stats)
            self.assertIn('Sharpe Ratio', stats)
            self.assertIn('# Trades', stats)
            self.assertIn('Max. Drawdown [%]', stats)

        # Verify averages
        avg_metrics = results['average_metrics']
        self.assertIn('Return [%]', avg_metrics)
        self.assertIn('Sharpe Ratio', avg_metrics)

if __name__ == "__main__":
    unittest.main()
