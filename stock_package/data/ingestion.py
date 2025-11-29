import yfinance as yf
import pandas as pd
from ..db.market_data import MarketDataDB
from ..config import MARKET_DATA_DB

class DataIngestion:
    def __init__(self, db_path=MARKET_DATA_DB):
        self.db_path = db_path

    def _store_data(self, db, symbol, hist):
        if not hist.empty:
            # Prepare data for insertion
            # Reset index to get Date/Datetime as column
            hist.reset_index(inplace=True)
            
            # Ensure columns exist and are in order: timestamp, open, high, low, close, volume
            # yfinance returns 'Datetime' for hourly data usually
            date_col = 'Datetime' if 'Datetime' in hist.columns else 'Date'
            
            data_to_insert = []
            for _, row in hist.iterrows():
                data_to_insert.append((
                    row[date_col].isoformat(),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ))
            
            db.add_prices(symbol, data_to_insert)
            print(f"Stored {len(data_to_insert)} records for {symbol}.")
        else:
            print(f"No data found for {symbol}.")

    def _update_stock_info(self, db, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            db.add_stock(
                symbol, 
                name=info.get('longName'), 
                sector=info.get('sector'), 
                industry=info.get('industry')
            )
            return ticker
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return None

    def fetch_and_initialise(self, symbols, period='1y', interval='1h'):
        """
        Initialise data for symbols. Fetches data for the specified period.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        with MarketDataDB(self.db_path) as db:
            for symbol in symbols:
                print(f"Initialising data for {symbol}...")
                ticker = self._update_stock_info(db, symbol)
                if ticker:
                    try:
                        hist = ticker.history(period=period, interval=interval)
                        self._store_data(db, symbol, hist)
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")

    def fetch_and_append(self, symbols, interval='1h'):
        """
        Append new data for symbols. Fetches data since the last stored timestamp.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        with MarketDataDB(self.db_path) as db:
            for symbol in symbols:
                print(f"Appending data for {symbol}...")
                ticker = self._update_stock_info(db, symbol)
                if ticker:
                    try:
                        last_ts = db.get_last_timestamp(symbol)
                        if last_ts:
                            import datetime
                            last_dt = pd.to_datetime(last_ts)
                            start_date = (last_dt + datetime.timedelta(hours=1)).strftime('%Y-%m-%d')
                            print(f"Found existing data for {symbol} up to {last_ts}. Fetching from {start_date}...")
                            
                            # Note: yfinance might return empty if start date is in the future or today
                            hist = ticker.history(start=start_date, interval=interval)
                            self._store_data(db, symbol, hist)
                        else:
                            print(f"No existing data for {symbol}. Use fetch_and_initialise first.")
                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
