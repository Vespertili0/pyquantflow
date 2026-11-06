import json
import logging
import os
import sqlite3
import numpy as np

logger = logging.getLogger(__name__)


class BacktestDatabaseManager:
    def __init__(self, db_path="backtest_results.db"):
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                batch_run_name TEXT NOT NULL,
                metrics JSON
            )
        """)
        self.conn.commit()

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save_result(self, ticker, result_dict, batch_run_name):
        """
        Saves a single backtest result to the database.

        Args:
            ticker (str): The ticker symbol.
            result_dict (dict): The dictionary containing backtest metrics.
            batch_run_name (str): The unique name of the batch run.
        """
        cursor = self.conn.cursor()

        clean_dict = {}
        for k, v in result_dict.items():
            if str(k).startswith("_"):
                continue
            if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                clean_dict[k] = None
            elif isinstance(v, np.integer):
                clean_dict[k] = int(v)
            elif isinstance(v, np.floating):
                clean_dict[k] = float(v)
            else:
                clean_dict[k] = v

        try:
            metrics_json = json.dumps(clean_dict, default=str, allow_nan=False)
        except Exception as e:
            logger.error(f"Error serializing results for {ticker}: {type(e).__name__}")
            return

        cursor.execute(
            """
            INSERT INTO backtest_results (ticker, batch_run_name, metrics)
            VALUES (?, ?, ?)
        """,
            (ticker, batch_run_name, metrics_json),
        )

        self.conn.commit()
