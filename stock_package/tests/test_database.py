from stock_package.data.database import DatabaseManager
import pandas as pd

def example_database_usage():
    print("Initializing Database Manager...")
    db = DatabaseManager("stocks.db")
    
    ticker = "BHP.AX"
    
    # 1. Add Ticker (or update if exists)
    print(f"\n--- Adding/Updating {ticker} ---")
    # Using a recent start year to make the initial fetch faster for demonstration
    db.add_ticker(ticker, start_year=2023)
    
    # 2. Retrieve Data
    print(f"\n--- Retrieving Data for {ticker} ---")
    df = db.get_data(ticker)
    print(f"Retrieved {len(df)} rows.")
    if not df.empty:
        print("Head:")
        print(df.head())
        print("Tail:")
        print(df.tail())
    
    # 3. Update Ticker (Simulate update)
    # Since we just added it, this might not find new data unless the market is open and we just crossed an hour mark.
    # But it verifies the logic runs without error.
    print(f"\n--- Updating {ticker} ---")
    db.update_ticker(ticker)
    
    # 4. Retrieve again to confirm
    df_updated = db.get_data(ticker)
    print(f"Retrieved {len(df_updated)} rows after update.")

if __name__ == "__main__":
    example_database_usage()
