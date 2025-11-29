from .base import BaseDB

class BacktestResultsDB(BaseDB):
    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                symbol TEXT,
                start_date DATETIME,
                end_date DATETIME,
                return_pct REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                trades INTEGER,
                duration TEXT
            )
        ''')
        self.conn.commit()

    def save_result(self, strategy_name, symbol, start_date, end_date, return_pct, sharpe_ratio, max_drawdown, trades, duration):
        self.cursor.execute('''
            INSERT INTO results (strategy_name, symbol, start_date, end_date, return_pct, sharpe_ratio, max_drawdown, trades, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (strategy_name, symbol, start_date, end_date, return_pct, sharpe_ratio, max_drawdown, trades, duration))
        self.conn.commit()
