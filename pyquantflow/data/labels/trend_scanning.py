import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from functools import partial

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["window"])
def _rolling_ols_t_stat_opt(prices, window):
    """
    Highly optimised rolling t-statistic calculation using analytical OLS
    and memory-efficient dynamic slicing.
    """
    n = prices.shape[0]
    num_windows = n - window + 1

    # Precompute analytical constants for X = [0, 1, ..., window-1]
    x = jnp.arange(window, dtype=jnp.float64)
    x_mean = (window - 1) / 2.0
    ss_xx = (window * (window**2 - 1)) / 12.0
    x_dev = x - x_mean

    # Vectorized function over a single window index using dynamic slicing
    def scan_window(start_idx):
        y = jax.lax.dynamic_slice_in_dim(prices, start_idx, window)
        y_mean = jnp.mean(y)

        # Analytical OLS components
        ss_yy = jnp.sum((y - y_mean) ** 2)
        beta1 = jnp.sum(x_dev * y) / ss_xx

        # Calculate SSR using standard OLS algebraic identity
        ssr = jnp.maximum(ss_yy - (beta1**2) * ss_xx, 1e-12)

        # Standard error and t-statistic
        sigma = jnp.sqrt(ssr / (window - 2))
        slope_se = sigma / jnp.sqrt(ss_xx)
        return beta1 / slope_se

    # Compute across all windows in parallel on-device
    t_stats = jax.vmap(scan_window)(jnp.arange(num_windows))
    return t_stats


def trend_scanning(
    series: pd.Series, windows: list | int = [5, 10, 20, 40, 80, 120]
) -> pd.DataFrame:
    """
    An optimised, memory-safe execution of Trend Scanning via JAX.
    """
    arr = jnp.array(series.values, dtype=jnp.float64)
    n = len(series)

    if isinstance(windows, int):
        windows = [windows]

    t_stats_collection = []

    # 1. Loop and process entirely on-device
    for w in windows:
        if w >= n:
            t_stats_collection.append(jnp.full(n, jnp.nan))
            continue

        t_vals = _rolling_ols_t_stat_opt(arr, w)

        # Pad with NaNs on-device using JAX instead of breaking out to NumPy
        padded = jnp.pad(t_vals, (0, n - t_vals.shape[0]), constant_values=jnp.nan)
        t_stats_collection.append(padded)

    # 2. Stack and bring back to Host memory only ONCE at the end
    all_t = np.array(jnp.stack(t_stats_collection, axis=1))
    abs_t = np.abs(all_t)

    valid_rows = ~np.isnan(abs_t).all(axis=1)
    final_values = np.full(n, np.nan)
    t1_times = pd.Series(pd.NaT, index=series.index, dtype=series.index.dtype)

    if np.any(valid_rows):
        valid_abs = abs_t[valid_rows]
        valid_raw = all_t[valid_rows]

        best_window_idx = np.nanargmax(valid_abs, axis=1)
        row_indices = np.arange(len(best_window_idx))

        final_values[valid_rows] = valid_raw[row_indices, best_window_idx]

        windows_arr = np.array(windows)
        chosen_windows = windows_arr[best_window_idx]

        full_row_indices = np.where(valid_rows)[0]
        end_indices = np.clip(full_row_indices + chosen_windows - 1, 0, n - 1)

        t1_times.iloc[valid_rows] = series.index[end_indices]

    return pd.DataFrame({"t_value": final_values, "t1": t1_times}, index=series.index)
