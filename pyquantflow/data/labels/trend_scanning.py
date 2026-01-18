import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from functools import partial

# Set JAX to 64-bit precision for financial calculations
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=['window'])
def _rolling_ols_t_stat(prices, window):
    """
    Computes rolling t-statistics of the slope (time trend) for a fixed window.
    """
    n = prices.shape[0]
    
    # Create the window indices: shape (n - window + 1, window)
    starts = jnp.arange(n - window + 1)
    indices = starts[:, None] + jnp.arange(window)[None, :]
    
    # Slice data: (num_windows, window_size)
    y = prices[indices]
    
    # X matrix is just time: [0, 1, ..., window-1]
    # We create a design matrix for [1, t] to handle intercept + slope
    x_i = jnp.arange(window, dtype=jnp.float64)
    X = jnp.stack([jnp.ones(window), x_i], axis=1) # (window, 2)
    
    # We want to solve (X^T X)^-1 X^T y for each window.
    # Since X is constant for all windows, we precompute (X^T X)^-1 X^T
    XtX_inv_Xt = jnp.linalg.inv(X.T @ X) @ X.T # (2, window)
    
    # Betas: (num_windows, 2) -> Intercept, Slope
    betas = jnp.dot(y, XtX_inv_Xt.T)
    
    # Calculate t-statistics
    # Residuals
    y_pred = jnp.dot(betas, X.T) # (num_windows, window)
    residuals = y - y_pred
    
    # Variance of residuals (SSR / (n - k))
    # k = 2 (slope + intercept)
    dof = window - 2
    ssr = jnp.sum(residuals**2, axis=1)
    sigma_sq = ssr / dof
    
    # Standard Error of slope
    # Covariance Matrix = sigma^2 * (X^T X)^-1
    # We need the (1,1) element of (X^T X)^-1 for the slope variance
    XtX_inv = jnp.linalg.inv(X.T @ X)
    slope_var = sigma_sq * XtX_inv[1, 1]
    slope_se = jnp.sqrt(slope_var)
    
    t_stats = betas[:, 1] / slope_se
    return t_stats

def trend_scanning(
    series: pd.Series, 
    windows: list | int = [5, 10, 20, 40, 80, 120]
    ) -> pd.Series:
    """
    Performs Trend Scanning by calculating the t-statistic of the slope 
    over multiple look-forward windows and selecting the one with the 
    maximum absolute t-statistic.
    
    Args:
        series (pd.Series): Price series.
        windows (list | int): A list of look-forward window sizes to scan (e.g. [10, 20, 60]).
                              If a single int is provided, it is treated as a list of one.
        
    Returns:
        pd.Series: The t-statistic of the trend from the most significant window.
    """
    # 1. Prepare Data
    arr = jnp.array(series.values, dtype=jnp.float64)
    n = len(series)
    
    # Handle single window case for backward compatibility
    if isinstance(windows, int):
        windows = [windows]
    
    t_stats_collection = []
    
    # 2. Compute JAX for each window
    for w in windows:
        if w >= n:
            # Window larger than data, fill with NaN
            t_stats_collection.append(np.full(n, np.nan))
            continue
            
        # Get raw JAX array (length = n - w + 1)
        t_vals = _rolling_ols_t_stat(arr, w)
        
        # Convert to numpy and Pad to length n
        # Because it's a look-forward, index i corresponds to [i, i+w]
        # So we align at 0, and the last w-1 elements are NaN
        padded = np.full(n, np.nan)
        padded[:len(t_vals)] = np.array(t_vals)
        t_stats_collection.append(padded)
    
    # 3. Stack and find Best Window
    # Shape: (n_samples, n_windows)
    all_t = np.stack(t_stats_collection, axis=1)
    
    # Calculate Absolute values for comparison, preserving NaNs
    abs_t = np.abs(all_t)
    
    # Mask rows where all are NaN (e.g., end of the series where no window fits)
    valid_rows = ~np.isnan(abs_t).all(axis=1)
    
    final_values = np.full(n, np.nan)
    
    if np.any(valid_rows):
        # Filter down to valid rows to avoid 'All-NaN slice' warning
        valid_abs = abs_t[valid_rows]
        valid_raw = all_t[valid_rows]
        
        # argmax of absolute values (ignoring NaNs in specific columns)
        best_window_idx = np.nanargmax(valid_abs, axis=1)
        
        # Extract corresponding signed t-values
        # Advanced indexing: [row_0, row_1...], [col_best_0, col_best_1...]
        row_indices = np.arange(len(best_window_idx))
        selected_values = valid_raw[row_indices, best_window_idx]
        
        final_values[valid_rows] = selected_values
    
    return pd.Series(final_values, index=series.index, name='t_value')