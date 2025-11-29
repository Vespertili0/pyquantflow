from .base import BaseDB

class MarketDataDB(BaseDB):
    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                FOREIGN KEY(symbol) REFERENCES stocks(symbol),
                UNIQUE(symbol, timestamp)
            )
        ''')
        self.conn.commit()

    def add_stock(self, symbol, name=None, sector=None, industry=None):
        self.cursor.execute('''
            INSERT OR IGNORE INTO stocks (symbol, name, sector, industry)
            VALUES (?, ?, ?, ?)
        ''', (symbol, name, sector, industry))
        self.conn.commit()

    def add_prices(self, symbol, data):
        # data is expected to be a list of tuples or a pandas DataFrame converted to records
        # (timestamp, open, high, low, close, volume)
        self.cursor.executemany('''
            INSERT OR IGNORE INTO prices (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [(symbol, *row) for row in data])
        self.conn.commit()
    
    def get_prices(self, symbol):
        self.cursor.execute('SELECT * FROM prices WHERE symbol = ? ORDER BY timestamp', (symbol,))
        return self.cursor.fetchall()

    def get_last_timestamp(self, symbol):
        self.cursor.execute('SELECT MAX(timestamp) FROM prices WHERE symbol = ?', (symbol,))
        result = self.cursor.fetchone()
        return result[0] if result else None
