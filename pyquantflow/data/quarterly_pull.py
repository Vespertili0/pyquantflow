import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_quarterly_data(ticker, time_dict, period="quarterly") -> pd.DataFrame:
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
    if period not in ["quarterly"]:
        raise ValueError("period must be 'quarterly'")

    all_period = {
        1: ("01-01", "04-01"),  # Q1 (end is exclusive in yfinance)
        2: ("04-01", "07-01"),  # Q2
        3: ("07-01", "10-01"),  # Q3
        4: ("10-01", "01-01"),  # Q4 (rolls into year + 1)
    }
    interval = "1h"
    data_frames = []

    for year, timeframes in time_dict.items():
        for t in timeframes:
            start, end = all_period[t]
            end_year = int(year) + 1 if t == 4 else int(year)
            try:
                new = yf.download(
                    tickers=ticker,
                    start=f"{year}-{start}",
                    end=f"{end_year}-{end}",
                    interval=interval,
                    multi_level_index=False,
                    auto_adjust=True,
                    progress=False,
                )
                if not new.empty:
                    data_frames.append(new)
            except Exception as e:
                logger.error(
                    f"Failed to fetch data for {year} Q{t}: {type(e).__name__}"
                )
                continue

    if not data_frames:
        return pd.DataFrame()

    data = pd.concat(data_frames)
    if data.empty:
        return data

    data = data[~data.index.duplicated(keep="first")].sort_index()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    return data
