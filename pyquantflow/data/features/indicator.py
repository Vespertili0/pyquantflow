import numpy as np
import pandas as pd


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
#    if close is not None:
#        close_s = pd.Series(close)
#        chikou_span = close_s.shift(-displacement).to_numpy()

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


def ROGERSATCHELL(high, low, open, close, timeperiod=30):
    """
    Rogers-Satchell Volatility for TA-Lib style function calls.
    Uses the 'cumsum' trick to achieve near-C speeds without Numba or JAX.

    Parameters
    ----------
    high, low, open, close : np.ndarray
        Input price arrays (float). Must be the same length.
    timeperiod : int
        The rolling window size (default 30).

    Returns
    -------
    np.ndarray
        Volatility array of the same length as inputs.
        The first `timeperiod` elements are NaN.
    """
    # 1. Input Validation (TA-Lib style strictness)
    # Ensure inputs are float arrays to prevent integer division errors
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    o = np.asarray(open, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    if not (h.shape == l.shape == o.shape == c.shape):
        raise ValueError("All input arrays must have the same shape.")

    n = h.shape[0]
    if timeperiod > n:
        # If data is shorter than window, return all NaNs
        return np.full(n, np.nan)

    # 2. Vectorized Math (Rogers-Satchell Formula)
    # rs = log(h/c)*log(h/o) + log(l/c)*log(l/o)
    # Use log(a/b) = log(a) - log(b) which is slightly safer/faster 
       
    # term1 = ln(High / Close) * ln(High / Open)
    term1 = np.log(h / c) * np.log(h / o)
    
    # term2 = ln(Low / Close) * ln(Low / Open)
    term2 = np.log(l / c) * np.log(l / o)
    
    rs_daily = term1 + term2

    # 3. The Optimization: Rolling Sum via Cumsum Trick
    # Sum[i:i+w] = CumSum[i+w] - CumSum[i]
    ret = np.cumsum(rs_daily, dtype=float)
    ret[timeperiod:] = ret[timeperiod:] - ret[:-timeperiod]
      
    # 4. Variance to Volatility
    # Divide by window size and sqrt
    vol = np.sqrt(ret / timeperiod)

    # 5. Padding
    # TA-Lib convention: if window is 30, indices 0..28 are NaN. Index 29 is the first valid value.
    # Usually TA-Lib returns NaN for indices 0 to timeperiod-2.
    vol[:timeperiod-1] = np.nan
    
    return vol