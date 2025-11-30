import yfinance as yf
import pandas as pd

def merge_last_hour(df)->pd.DataFrame:
    """
    Merges the last trading hour of each day into the previous hour.
    """
    df = df.sort_index()
    df = df.copy()

    # Group by date to handle each day separately
    for date, group in df.groupby(df.index.date):
        if len(group) < 2:
            continue  # Skip if there's no previous hour to merge with

        last_idx = group.index[-1]
        prev_idx = group.index[-2]

        # Merge logic
        df.at[prev_idx, 'High'] = max(df.at[prev_idx, 'High'], df.at[last_idx, 'High'])
        df.at[prev_idx, 'Low'] = min(df.at[prev_idx, 'Low'], df.at[last_idx, 'Low'])
        df.at[prev_idx, 'Close'] = df.at[last_idx, 'Close']
        df.at[prev_idx, 'Volume'] += df.at[last_idx, 'Volume']

        # Drop the last row
        df = df.drop(last_idx)

    return df


def fetch_quarterly_data(ticker, time_dict, period='quarterly')->pd.DataFrame:
    """
    Fetches stock data from Yahoo Finance for specified quarters and years, bypassing
    API limitations by breaking down requests into smaller time frames.

    Parameters:
    ticker (str): The stock ticker symbol.
    year_quarters (dict): A dictionary where keys are years (in 'yyyy' format)
                          and values are lists of quarter numbers (1, 2, 3, 4).

    Returns:
    pd.DataFrame: A concatenated DataFrame containing the data for all selected quarters.
    """
    assert period in ['quarterly'], 'period must be quarterly or bimonthly'
    # Define all quarters with their start and end dates
    if period == 'quarterly':
        all_period = {
            1: ("01-01", "03-31"),  # Q1
            2: ("04-01", "06-30"),  # Q2
            3: ("07-01", "09-30"),  # Q3
            4: ("10-01", "12-31")   # Q4
        }
        interval = '1h'
    data = pd.DataFrame()  # To store the final concatenated data

    # Iterate through the years and their respective quarters
    for year, timeframes in time_dict.items():
        for t in timeframes:
            start, end = all_period[t]
            try:
                # Download data for the specific quarter
                new = yf.download(
                    tickers=ticker,
                    start=f"{year}-{start}",
                    end=f"{year}-{end}",
                    interval=interval,
                    multi_level_index=False,
                    auto_adjust=True,
                    progress=False,  # Suppress the progress bar
                )
                # Concatenate the new data
                data = pd.concat([data, new])
            except Exception as e:
                print(f"Failed to fetch data for {year} Q{t}: {e}")
                break
    data.index = data.index.tz_convert('Australia/Sydney')
#    data = merge_last_hour(data)
#    data = data.between_time("11:00", "16:00")
#    data = data[data.Volume > 0]
    return data