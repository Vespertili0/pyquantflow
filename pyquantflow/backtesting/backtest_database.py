import sqlite3
import json
from datetime import datetime

class BacktestDatabaseManager:
    def __init__(self, db_path="backtest_results.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metrics JSON
            )
        """)
        self.conn.commit()

    def save_result(self, ticker, result_dict):
        """
        Saves a single backtest result to the database.
        
        Args:
            ticker (str): The ticker symbol.
            result_dict (dict): The dictionary containing backtest metrics.
        """
        cursor = self.conn.cursor()
        
        # Serialize metrics to JSON
        # Handle potential non-serializable types if necessary (though backtesting.py results are mostly float/str)
        # If there are Timedeltas or NaNs, simple json.dumps might fail or produce invalid JSON.
        # We'll use a custom encoder or string conversion for safety if needed, 
        # but for now standard dumps with default=str should cover most cases (like Timedelta).
        try:
            metrics_json = json.dumps(result_dict, default=str)
        except Exception as e:
            print(f"Error serializing results for {ticker}: {e}")
            return

        cursor.execute("""
            INSERT INTO backtest_results (ticker, run_date, metrics)
            VALUES (?, ?, ?)
        """, (ticker, datetime.now(), metrics_json))
        
        self.conn.commit()
        # print(f"Saved results for {ticker} to DB.")
