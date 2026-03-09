import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
from .quarterly_pull import fetch_quarterly_data

class DatabaseManager:
    def __init__(self, db_path="stocks.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                interval TEXT DEFAULT '1h',
                first_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP
            )
        """)

        # Check if interval column exists (for migration)
        cursor.execute("PRAGMA table_info(tickers)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'interval' not in columns:
            try:
                cursor.execute("ALTER TABLE tickers ADD COLUMN interval TEXT DEFAULT '1h'")
            except sqlite3.OperationalError:
                # Column might have been added concurrently
                pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_id INTEGER,
                datetime TIMESTAMP,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                FOREIGN KEY(ticker_id) REFERENCES tickers(id)
            )
        """)
        self.conn.commit()

    def add_ticker(self, ticker, start_date=None, start_year=None, interval='1h'):
        """
        Adds a new ticker to the database.
        Fetches historical data using quarterly_pull (for 1h) or direct download (for 1d).
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM tickers WHERE ticker = ?", (ticker,))
        if cursor.fetchone():
            print(f"Ticker {ticker} already exists. Updating instead.")
            self.update_ticker(ticker)
            return

        # Determine start date
        if start_date:
            # yfinance expects string YYYY-MM-DD or datetime
            start = start_date
            try:
                # If start_date is string, parse to get year for quarterly fallback logic
                if isinstance(start_date, str):
                    start_year_val = int(start_date.split('-')[0])
                else:
                    start_year_val = start_date.year
            except Exception:
                start_year_val = 2020
        else:
            if start_year is None:
                start_year = 2020
            start_year_val = start_year
            start = f"{start_year}-01-01"

        print(f"Fetching initial data for {ticker} from {start} (interval={interval})...")

        df = pd.DataFrame()
        try:
            if interval == '1d':
                # For daily data, we can download in one go
                df = yf.download(ticker, start=start, interval='1d', progress=False, auto_adjust=True, multi_level_index=False)
            else:
                # Default 1h behavior using quarterly_pull
                # Generate time_dict for quarterly pull
                current_year = datetime.now().year
                time_dict = {}
                for year in range(start_year_val, current_year + 1):
                    time_dict[year] = [1, 2, 3, 4]

                df = fetch_quarterly_data(ticker, time_dict)
        except TypeError:
             # Fallback for yfinance versions without multi_level_index
             if interval == '1d':
                 df = yf.download(ticker, start=start, interval='1d', progress=False, auto_adjust=True)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return
        
        if df.empty:
            print(f"No data found for {ticker}")
            return

        # Insert ticker
        cursor.execute("INSERT INTO tickers (ticker, interval, last_updated) VALUES (?, ?, ?)", (ticker, interval, datetime.now()))
        ticker_id = cursor.lastrowid

        # Insert data
        self._insert_price_data(ticker_id, df)
        self.conn.commit()
        print(f"Added {ticker} with {len(df)} records.")

    def update_ticker(self, ticker):
        """
        Updates an existing ticker with new data since the last entry.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, last_updated, interval FROM tickers WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if not row:
            print(f"Ticker {ticker} not found.")
            return
        
        ticker_id = row[0]
        interval = row[2] if len(row) > 2 else '1h'
        
        # Get last datetime from price_data
        cursor.execute("SELECT MAX(datetime) FROM price_data WHERE ticker_id = ?", (ticker_id,))
        last_date_str = cursor.fetchone()[0]
        
        if not last_date_str:
            self.add_ticker(ticker)
            return

        # Convert to datetime (handling potential timezone info in string)
        # SQLite stores as string. Pandas to_datetime is smart.
        last_date = pd.to_datetime(last_date_str)
        
        # yfinance expects naive or localized? 
        # If last_date has tz, we should probably keep it.
        # But yf.download start parameter expects string 'YYYY-MM-DD' or datetime.
        
        print(f"Updating {ticker} from {last_date} (interval={interval})...")
        
        # Download new data
        try:
            new_data = yf.download(ticker, start=last_date, interval=interval, progress=False, auto_adjust=True, multi_level_index=False)
        except TypeError:
            # Fallback for older versions of yfinance that don't support multi_level_index
            new_data = yf.download(ticker, start=last_date, interval=interval, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"Error updating data: {e}")
            return
        
        if new_data.empty:
            print("No new data.")
            return

        # Filter to ensure we only get strictly newer data
        # If last_date is tz-aware, new_data index should be too.
        if new_data.index.tz is None and last_date.tz is not None:
             new_data.index = new_data.index.tz_localize(last_date.tz)
        elif new_data.index.tz is not None and last_date.tz is None:
             last_date = last_date.tz_localize(new_data.index.tz)

        new_data = new_data[new_data.index > last_date]
        
        if new_data.empty:
            print("No new data after filtering.")
            return

        # Convert timezone to match quarterly_pull (Australia/Sydney) if needed
        # quarterly_pull does: data.index = data.index.tz_convert('Australia/Sydney')
        # We should probably maintain consistency.
        try:
            new_data.index = new_data.index.tz_convert('Australia/Sydney')
        except Exception:
            pass # If conversion fails (e.g. naive), skip or handle

        self._insert_price_data(ticker_id, new_data)
        
        cursor.execute("UPDATE tickers SET last_updated = ? WHERE id = ?", (datetime.now(), ticker_id))
        self.conn.commit()
        print(f"Updated {ticker} with {len(new_data)} new records.")

    def _insert_price_data(self, ticker_id, df):
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        df_reset = df.reset_index()
        data_to_insert = []
        for _, row in df_reset.iterrows():
            # Identify date column
            dt_col = 'Datetime' if 'Datetime' in df_reset.columns else 'Date'
            if dt_col not in df_reset.columns:
                # Fallback, maybe index wasn't named?
                # After reset_index, the first column is usually the index.
                dt = row.iloc[0]
            else:
                dt = row[dt_col]

            data_to_insert.append((
                ticker_id,
                str(dt),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            ))
        
        self.conn.executemany("""
            INSERT INTO price_data (ticker_id, datetime, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data_to_insert)

    def get_data(self, ticker):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM tickers WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if not row:
            print(f"Ticker {ticker} not found in database.")
            return pd.DataFrame()
        
        ticker_id = row[0]
        query = "SELECT datetime, open, high, low, close, volume FROM price_data WHERE ticker_id = ? ORDER BY datetime"
        df = pd.read_sql_query(query, self.conn, params=(ticker_id,), parse_dates={'datetime': {'utc': True}})
        df = df.set_index('datetime')
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        return df
