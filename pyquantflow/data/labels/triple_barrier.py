import numpy as np
import pandas as pd

def apply_triple_barrier(prices, sl_col, tp_mult, horizon):
    """
    prices: pd.Series of prices
    sl_col: pd.Series of stop-loss levels (absolute price levels)
    tp_mult: float, multiplier applied to distance between price and SL to compute TP
    horizon: int, number of bars before timeout

    Returns: pd.Series of labels {0: SL hit, 1: timeout, 2: TP hit}
    """
    n = len(prices)
    prices_arr = prices.values
    sl_arr = sl_col.values

    # Compute TP barrier
    tp_arr = prices_arr + tp_mult * (prices_arr - sl_arr)

    # Preallocate labels
    labels = np.full(n, np.nan)

    # Compute end index for each row
    end_idx = np.minimum(np.arange(n) + horizon, n - 1)
    full_window = (np.arange(n) + horizon < n)

    # Build index matrix safely
    max_h = horizon + 1
    offsets = np.arange(max_h)

    # valid_mask[i, j] = True if i+j <= end_idx[i]
    valid_mask = offsets[None, :] <= (end_idx - np.arange(n))[:, None]

    # Use np.where to avoid invalid indices becoming 0
    idx_matrix = np.where(
        valid_mask,
        np.arange(n)[:, None] + offsets[None, :],
        -1  # invalid index
    )

    # Gather forward prices, ignoring invalid entries
    forward_prices = np.where(
        idx_matrix >= 0,
        prices_arr[idx_matrix],
        np.nan
    )

    # Barrier hits
    sl_hit = (forward_prices <= sl_arr[:, None]).any(axis=1)
    tp_hit = (forward_prices >= tp_arr[:, None]).any(axis=1)

    labels[sl_hit] = 0
    labels[tp_hit] = 2

    # Timeout only if full window exists AND no hit
    timeout_mask = (~sl_hit) & (~tp_hit) & full_window
    labels[timeout_mask] = 1

    return pd.Series(labels, index=prices.index)