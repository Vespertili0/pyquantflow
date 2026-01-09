import numpy as np
import pandas as pd


def pipe_indicator(df: pd.DataFrame, indicator, input_map, output_names, **kwargs) -> pd.DataFrame:
    """
    A Pandas pipe-compatible function to calculate indicators and inject them 
    back into the DataFrame.

    Args:
        df (pd.DataFrame): The input dataframe.
        func (callable): The indicator function (e.g., ICHIMOKU or talib.RSI).
        input_map (dict or list): 
            - If dict: maps function arguments to DF column names. e.g. {'high': 'High', 'low': 'Low'}
            - If list: maps positional function arguments to DF column names. e.g. ['Close']
        output_names (str or list): Names for the resulting columns. 
            - If the function returns a tuple, provide a list of names. 
            - Use None in the list to skip specific return values.
        **kwargs: Static arguments passed to the indicator function (e.g., timeperiod=14).

    Returns:
        pd.DataFrame: The dataframe with new indicator columns.
    """
    
    # 1. Prepare Data Inputs
    if isinstance(input_map, dict):
        # Pass data as Keyword Arguments (Good for functions with named inputs like ours)
        data_inputs = {arg: df[col].values for arg, col in input_map.items()}
        # Combine with static kwargs
        full_kwargs = {**data_inputs, **kwargs}
        results = indicator(**full_kwargs)
        
    elif isinstance(input_map, list) or isinstance(input_map, tuple):
        # Pass data as Positional Arguments (Good for standard TA-Lib functions like RSI)
        pos_inputs = [df[col].values for col in input_map]
        results = indicator(*pos_inputs, **kwargs)
    else:
        raise ValueError("input_map must be a dict or list/tuple")

    # 2. Handle Output Assignment
    
    # Normalize results to be iterable if it's a single value
    if not isinstance(results, tuple):
        results = (results,)
    
    # Normalize output_names to list
    if isinstance(output_names, str):
        output_names = [output_names]

    # Assign columns
    for name, data in zip(output_names, results):
        if name is not None and data is not None:
            df[name] = data
            
    return df


def ICHIMOKU(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray = None, 
    tenkan_period: int = 9, 
    kijun_period: int = 26, 
    senkou_b_period: int = 52,
    displacement: int = 26
) -> tuple:
    """
    Computes Ichimoku Cloud elements in a TA-Lib style.
    
    Args:
        high (np.ndarray or pd.Series): High prices.
        low (np.ndarray or pd.Series): Low prices.
        close (np.ndarray or pd.Series, optional): Close prices (needed for Chikou Span).
        tenkan_period (int): Period for Conversion Line (default 9).
        kijun_period (int): Period for Base Line (default 26).
        senkou_b_period (int): Period for Leading Span B (default 52).
        displacement (int): Displacement for Spans/Chikou (default 26).

    Returns:
        tuple: A tuple containing the following numpy arrays:
            (
                tenkan_sen,      # Conversion Line
                kijun_sen,       # Base Line
                span_a,          # Leading Span A (Projected value recorded at current time)
                span_b,          # Leading Span B (Projected value recorded at current time)
                span_a_shifted,  # Leading Span A (Shifted forward to align with current price)
                span_b_shifted,  # Leading Span B (Shifted forward to align with current price)
                chikou_span      # Lagging Span (Close shifted backwards) - None if close not provided
            )
    """
    
    # Convert to Pandas Series for efficient rolling window calculations
    # This handles both numpy array and pandas series inputs gracefully
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # --- 1. Tenkan-sen (Conversion Line) ---
    tenkan_high = high_s.rolling(window=tenkan_period).max()
    tenkan_low = low_s.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # --- 2. Kijun-sen (Base Line) ---
    kijun_high = high_s.rolling(window=kijun_period).max()
    kijun_low = low_s.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2

    # --- 3. Senkou Span A (Leading Span A) ---
    # Recorded at current time t (Projected)
    span_a = (tenkan_sen + kijun_sen) / 2

    # --- 4. Senkou Span B (Leading Span B) ---
    # Recorded at current time t (Projected)
    span_b_high = high_s.rolling(window=senkou_b_period).max()
    span_b_low = low_s.rolling(window=senkou_b_period).min()
    span_b = (span_b_high + span_b_low) / 2

    # --- 5. Shifted Spans (The "Current Cloud") ---
    # Shifted forward to align with current price candle
    span_a_shifted = span_a.shift(displacement)
    span_b_shifted = span_b.shift(displacement)

    # --- 6. Chikou Span (Lagging Span) ---
    chikou_span = None
    if close is not None:
        close_s = pd.Series(close)
        chikou_span = close_s.shift(-displacement).to_numpy()

    # Return tuple of numpy arrays (TA-Lib style)
    return (
        tenkan_sen.to_numpy(),
        kijun_sen.to_numpy(),
        span_a.to_numpy(),
        span_b.to_numpy(),
        span_a_shifted.to_numpy(),
        span_b_shifted.to_numpy(),
        chikou_span
    )