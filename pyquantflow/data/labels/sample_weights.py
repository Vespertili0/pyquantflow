import numpy as np
import pandas as pd


def get_sample_weights(t1, returns=None):
    """
    Calculates sample uniqueness (u_i) and optionally weights it by absolute returns.

    Parameters:
    -----------
    t1 : pd.Series
        A Pandas Series where the index is the start time (t0) and the
        values are the end times (t1). Must be sorted chronologically.
    returns : pd.Series, optional
        A Pandas Series of returns corresponding to the same index.
        If provided, the final weight is Uniqueness * |Return|.

    Returns:
    --------
    pd.Series
        The final sample weights for each event.
    """
    # Normalize both index and values to UTC, then drop tz
    if hasattr(t1.index, "tz"):
        t1.index = t1.index.tz_localize("UTC") if t1.index.tz is None else t1.index
        t1.index = t1.index.tz_convert(None)

    if hasattr(t1, "dt"):
        t1 = t1.dt.tz_localize("UTC") if t1.dt.tz is None else t1
        t1 = t1.dt.tz_convert(None)

    # Ensure chronological order
    t1 = t1.sort_index()
    bars = t1.index.values

    # 1. Map timestamps to integer array indices
    start_idx = np.arange(len(bars))
    # Find where the end time falls in the bar index
    end_idx = np.searchsorted(bars, t1.values, side='right')

    # 2. Fast Concurrency (c_t) via Difference Array
    # Add +1 when an event starts, -1 when it ends
    diff = np.zeros(len(bars) + 1)
    for s, e in zip(start_idx, end_idx):
        diff[s] += 1
        diff[min(e, len(bars))] -= 1  # Cap at the end of our dataset

    # Cumulative sum gives the exact number of overlapping labels at each bar
    c_t = np.cumsum(diff)[:-1]

    # 3. Point Uniqueness (1 / c_t)
    point_uniqueness = np.zeros_like(c_t, dtype=float)
    np.divide(1.0, c_t, out=point_uniqueness, where=c_t > 0)

    # 4. Average Uniqueness per Sample (u_i)
    # We use a cumulative sum array of point_uniqueness to allow O(1) range sums
    cum_u = np.insert(np.cumsum(point_uniqueness), 0, 0)

    u_i = np.zeros(len(bars))
    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        e_clipped = min(e, len(bars))
        count = e_clipped - s

        if count > 0:
            # Sum of point uniqueness in range [s, e_clipped) divided by lifespan
            u_i[i] = (cum_u[e_clipped] - cum_u[s]) / count
        else:
            u_i[i] = 0.0

    # 5. Combine with Returns (Optional but highly recommended)
    weights = pd.Series(u_i, index=t1.index, name='weight')

    if returns is not None:
        # Align returns just in case, then multiply uniqueness by absolute return
        abs_rets = returns.reindex(t1.index).abs()
        weights = weights * abs_rets

    return weights