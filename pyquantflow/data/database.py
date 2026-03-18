import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from .quarterly_pull import fetch_quarterly_data

logger = logging.getLogger(__name__)

class DatabaseManager:
    _DEFAULT_INTERVAL = '1h'
    _DEFAULT_START_YEAR = 2020
    _TZ_DEFAULT = 'Australia/Sydney'

    _SQL_INSERT_TICKER = "INSERT INTO tickers (ticker, interval, last_updated) VALUES (?, ?, ?)"
    _SQL_UPDATE_TICKER_TIME = "UPDATE tickers SET last_updated = ? WHERE id = ?"
    _SQL_SELECT_TICKER = "SELECT id, last_updated, interval FROM tickers WHERE ticker = ?"
    _SQL_SELECT_MAX_DATETIME = "SELECT MAX(datetime) FROM price_data WHERE ticker_id = ?"
    _SQL_INSERT_PRICE_DATA = """
        INSERT INTO price_data (ticker_id, datetime, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    _SQL_SELECT_PRICE_DATA = """
        SELECT datetime, open, high, low, close, volume 
        FROM price_data 
        WHERE ticker_id = ? 
        ORDER BY datetime
    """

    def __init__(self, db_path="stocks.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")

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
                cursor.execute(
                    "ALTER TABLE tickers ADD COLUMN interval TEXT DEFAULT '1h'"
                )
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

    def add_ticker(
        self, 
        ticker, 
        start_date=None, 
        start_year=None, 
        interval=None, 
        commit=True
    ):
        """
        Adds a new ticker to the database.
        Fetches historical data using quarterly_pull (for 1h) or 
        direct download (for 1d).
        """
        interval = interval or self._DEFAULT_INTERVAL
        
        cursor = self.conn.cursor()
        cursor.execute(self._SQL_SELECT_TICKER, (ticker,))
        row = cursor.fetchone()
        
        if row:
            logger.info(f"Ticker {ticker} already exists. Updating instead.")
            # Pass ticker_id and interval to avoid redundant queries
            self._update_ticker_internal(ticker, row[0], row[2], commit=commit)
            return

        # Determine start date
        if start_date:
            start = start_date
            try:
                if isinstance(start_date, str):
                    start_year_val = int(start_date.split('-')[0])
                else:
                    start_year_val = start_date.year
            except Exception:
                start_year_val = self._DEFAULT_START_YEAR
        else:
            if start_year is None:
                start_year = self._DEFAULT_START_YEAR
            start_year_val = start_year
            start = f"{start_year}-01-01"

        logger.info(f"Fetching initial data for {ticker} from {start} (interval={interval})...")

        df = pd.DataFrame()
        try:
            if interval == '1d':
                df = yf.download(
                    ticker, 
                    start=start, 
                    interval='1d', 
                    progress=False, 
                    auto_adjust=True, 
                    multi_level_index=False
                )
            else:
                current_year = datetime.now().year
                time_dict = {year: [1, 2, 3, 4] for year in range(start_year_val, current_year + 1)}
                df = fetch_quarterly_data(ticker, time_dict)
        except TypeError:
             if interval == '1d':
                 df = yf.download(
                    ticker, 
                    start=start, 
                    interval='1d', 
                    progress=False, 
                    auto_adjust=True
                )
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return

        cursor.execute(self._SQL_INSERT_TICKER, (ticker, interval, datetime.now()))
        ticker_id = cursor.lastrowid

        self._insert_price_data(ticker_id, df)

        if commit:
            self.conn.commit()

        logger.info(f"Added {ticker} with {len(df)} records.")

    def update_ticker(self, ticker, commit=True):
        """
        Updates an existing ticker with new data since the last entry.
        """
        cursor = self.conn.cursor()
        cursor.execute(self._SQL_SELECT_TICKER, (ticker,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Ticker {ticker} not found. Adding instead.")
            self.add_ticker(ticker, commit=commit)
            return
        
        self._update_ticker_internal(ticker, row[0], row[2], commit=commit)

    def _update_ticker_internal(self, ticker, ticker_id, interval, commit=True):
        """
        Internal method to update a ticker, avoiding redundant database queries.
        """
        interval = interval or self._DEFAULT_INTERVAL
        cursor = self.conn.cursor()
        
        cursor.execute(self._SQL_SELECT_MAX_DATETIME, (ticker_id,))
        last_date_str = cursor.fetchone()[0]
        
        if not last_date_str:
            logger.warning(f"No previous data found for {ticker}. Attempting fresh fetch.")
            # Data was missing, fetch fresh data using default bounds
            # For simplicity, we just trigger yfinance or fallback to an empty df
            try:
                df = yf.download(
                    ticker, 
                    interval=interval, 
                    progress=False, 
                    auto_adjust=True,
                    multi_level_index=False
                )
            except Exception as e:
                logger.error(f"Error establishing baseline for {ticker}: {e}")
                return
            
            if not df.empty:
                self._insert_price_data(ticker_id, df)
                cursor.execute(self._SQL_UPDATE_TICKER_TIME, (datetime.now(), ticker_id))
                if commit:
                    self.conn.commit()
                logger.info(f"Updated {ticker} with {len(df)} new records.")
            return

        last_date = pd.to_datetime(last_date_str)
        
        logger.info(f"Updating {ticker} from {last_date} (interval={interval})...")
        
        try:
            new_data = yf.download(
                ticker, 
                start=last_date, 
                interval=interval, 
                progress=False, 
                auto_adjust=True, 
                multi_level_index=False
            )
        except TypeError:
            new_data = yf.download(
                ticker, 
                start=last_date, 
                interval=interval, 
                progress=False, 
                auto_adjust=True
            )
        except Exception as e:
            logger.error(f"Error updating data for {ticker}: {e}")
            return
        
        if new_data.empty:
            logger.info(f"No new data for {ticker}.")
            return

        # Filter
        if new_data.index.tz is None and last_date.tz is not None:
             new_data.index = new_data.index.tz_localize(last_date.tz)
        elif new_data.index.tz is not None and last_date.tz is None:
             last_date = last_date.tz_localize(new_data.index.tz)

        new_data = new_data[new_data.index > last_date]
        
        if new_data.empty:
            logger.info(f"No new data after filtering for {ticker}.")
            return

        try:
            new_data.index = new_data.index.tz_convert(self._TZ_DEFAULT)
        except TypeError:
            if new_data.index.tz is None:
                new_data.index = new_data.index.tz_localize('UTC').tz_convert(self._TZ_DEFAULT)
        except Exception as e:
            logger.warning(f"Timezone conversion failed for {ticker}. Proceeding with current index. Error: {e}")

        self._insert_price_data(ticker_id, new_data)
        
        cursor.execute(self._SQL_UPDATE_TICKER_TIME, (datetime.now(), ticker_id))

        if commit:
            self.conn.commit()

        logger.info(f"Updated {ticker} with {len(new_data)} new records.")

    def update_tickers_batch(self, tickers_list):
        """
        Updates multiple tickers and commits once at the end.
        """
        for ticker in tickers_list:
            try:
                self.update_ticker(ticker, commit=False)
            except Exception as e:
                logger.error(f"Error updating {ticker} in batch: {e}")

        self.conn.commit()
        logger.info(f"Batch update completed for {len(tickers_list)} tickers.")

    def _insert_price_data(self, ticker_id, df):
        if df.empty:
            return
            
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        df_reset = df.reset_index()
        
        dt_col = 'Datetime' if 'Datetime' in df_reset.columns else ('Date' if 'Date' in df_reset.columns else df_reset.columns[0])
        
        # Prepare data efficiently using itertuples (vectorized extraction)
        # Avoids the slow row-by-row iterrows() loop
        df_to_save = pd.DataFrame()
        df_to_save['ticker_id'] = [ticker_id] * len(df_reset)
        df_to_save['datetime'] = df_reset[dt_col].astype(str)
        df_to_save['open'] = df_reset['Open']
        df_to_save['high'] = df_reset['High']
        df_to_save['low'] = df_reset['Low']
        df_to_save['close'] = df_reset['Close']
        df_to_save['volume'] = df_reset['Volume']

        data_to_insert = list(df_to_save.itertuples(index=False, name=None))
        
        self.conn.executemany(self._SQL_INSERT_PRICE_DATA, data_to_insert)

    def get_data(self, ticker):
        cursor = self.conn.cursor()
        cursor.execute(self._SQL_SELECT_TICKER, (ticker,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Ticker {ticker} not found in database.")
            return pd.DataFrame()
        
        ticker_id = row[0]
        df = pd.read_sql_query(
            self._SQL_SELECT_PRICE_DATA,
            self.conn,
            params=(ticker_id,),
            parse_dates={'datetime': {'utc': True}}
        )
        
        if not df.empty:
            df = df.set_index('datetime')
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
        return df
