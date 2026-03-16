import pandas as pd
import numpy as np


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Applies Fixed-Width Window Fractional Differentiation.
    
    Args:
        series (pd.Series): Time series of prices (or log prices).
        d (float): The differencing order (e.g., 0.4).
        thres (float): Threshold for weight cutoff.
        
    Returns:
        pd.Series: Fractionally differentiated series.
    """
    # 1. Prepare Data
    arr = np.array(series.values, dtype=np.float64)
    
    # 2. Calculate Weights (Iterative logic done in numpy for dynamic sizing)
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    weights = np.array(w)
    
    # 3. Apply via pure NumPy convolution
    res = np.convolve(arr, weights[::-1], mode='full')
    
    # 4. Sandwich back to Pandas
    # The 'full' mode returns an array of size len(arr) + len(weights) - 1.
    # To match standard behavior where the index aligns with the end of the window:
    # First, discard the tail elements beyond the original series length.
    result = res[:len(arr)]
    
    # The first len(weights) - 1 elements are technically not valid because
    # they didn't have a full window. We assign NaN.
    result[:len(weights) - 1] = np.nan
    
    return pd.Series(result, index=series.index, name=f"frac_diff_{d}")