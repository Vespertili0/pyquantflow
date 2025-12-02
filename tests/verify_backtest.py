import pandas as pd
import sqlite3
import json
from stock_package.backtest_framework import BatchBacktester
from stock_package.strategies.example_strategy import SmaCross
from stock_package.backtesting.backtest_database import BacktestDatabaseManager

def verify_backtest():
    # 1. Setup dummy data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': [100 + i for i in range(100)],
        'High': [105 + i for i in range(100)],
        'Low': [95 + i for i in range(100)],
        'Close': [102 + i for i in range(100)],
        'Volume': [1000 for _ in range(100)]
    }, index=dates)
    
    ticker = "TEST_TICKER"
    data_map = {ticker: df}
    
    # 2. Run Backtest
    print("Running backtest...")
    backtester = BatchBacktester(results_db_path="test_results.db")
    results = backtester.run_batch_backtest(data_map, SmaCross)
    
    # 3. Verify Results Returned
    if ticker in results['individual_results']:
        print("SUCCESS: Backtest results returned.")
    else:
        print("FAILURE: No results returned.")
        return

    # 4. Verify Database Storage
    print("Verifying database storage...")
    db = BacktestDatabaseManager(db_path="test_results.db")
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM backtest_results WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    
    if row:
        print("SUCCESS: Results found in database.")
        # Verify JSON content
        stored_metrics = json.loads(row[3])
        if 'Return [%]' in stored_metrics:
             print(f"SUCCESS: Metrics verified. Return: {stored_metrics['Return [%]']:.2f}%")
        else:
             print("FAILURE: Metrics JSON structure incorrect.")
    else:
        print("FAILURE: No results found in database.")

    # Clean up
    import os
    if os.path.exists("test_results.db"):
        os.remove("test_results.db")

if __name__ == "__main__":
    verify_backtest()
