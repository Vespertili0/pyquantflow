import numpy as np
import pandas as pd

def apply_triple_barrier(prices, sl_col, tp_mult, horizon):
    """
    prices: pd.Series of prices (Index must be Datetime)
    sl_col: pd.Series of stop-loss levels (absolute price levels)
    tp_mult: float, multiplier applied to distance between price and SL to compute TP
    horizon: int, number of bars before timeout

    Returns: pd.DataFrame with ['label', 't1']
             label: {0: SL hit, 1: timeout, 2: TP hit}
             t1: Datetime when the barrier was resolved
    """
    n = len(prices)
    prices_arr = prices.values
    sl_arr = sl_col.values
    tp_arr = prices_arr + tp_mult * (prices_arr - sl_arr)

    # 1. Build index matrix 
    max_h = horizon + 1
    offsets = np.arange(max_h)
    end_idx = np.minimum(np.arange(n) + horizon, n - 1)
    
    valid_mask = offsets[None, :] <= (end_idx - np.arange(n))[:, None]
    idx_matrix = np.where(valid_mask, np.arange(n)[:, None] + offsets[None, :], -1)

    forward_prices = np.where(idx_matrix >= 0, prices_arr[idx_matrix], np.nan)

    # 2. Find the EXACT index offset of the first hit
    sl_mask = forward_prices <= sl_arr[:, None]
    tp_mask = forward_prices >= tp_arr[:, None]

    # np.argmax returns the first True index. (Returns 0 if all are False)
    sl_hit_idx = np.argmax(sl_mask, axis=1)
    tp_hit_idx = np.argmax(tp_mask, axis=1)

    # Track if a hit actually occurred in the window
    sl_hit_occurred = sl_mask.any(axis=1)
    tp_hit_occurred = tp_mask.any(axis=1)

    # If no hit occurred, set to infinity so we can cleanly compare which hit first
    sl_first_touch = np.where(sl_hit_occurred, sl_hit_idx, np.inf)
    tp_first_touch = np.where(tp_hit_occurred, tp_hit_idx, np.inf)

    # 3. Assign Labels and Resolve Conflicts
    labels = np.full(n, np.nan)
    t1_offsets = np.full(n, np.nan) # Stores the steps taken to resolve

    # Condition A: Stop Loss hit first (or tied with TP - be conservative!)
    sl_wins = sl_hit_occurred & (sl_first_touch <= tp_first_touch)
    labels[sl_wins] = 0
    t1_offsets[sl_wins] = sl_first_touch[sl_wins]

    # Condition B: Take Profit hit first
    tp_wins = tp_hit_occurred & (tp_first_touch < sl_first_touch)
    labels[tp_wins] = 2
    t1_offsets[tp_wins] = tp_first_touch[tp_wins]

    # Condition C: Timeout (Vertical Barrier)
    full_window = (np.arange(n) + horizon < n)
    timeout_mask = (~sl_hit_occurred) & (~tp_hit_occurred) & full_window
    labels[timeout_mask] = 1
    t1_offsets[timeout_mask] = horizon 

    # 4. Calculate actual t1 timestamps
    base_idx = np.arange(n)
    final_idx = base_idx + t1_offsets
    
    t1_times = pd.Series(pd.NaT, index=prices.index, dtype=prices.index.dtype)
    
    # Map valid integer indices back to the price index timestamps
    valid_t1 = ~np.isnan(final_idx) & (final_idx < n)
    t1_times.iloc[valid_t1] = prices.index[final_idx[valid_t1].astype(int)]

    return pd.DataFrame({
        'label': labels,
        't1': t1_times
    }, index=prices.index)