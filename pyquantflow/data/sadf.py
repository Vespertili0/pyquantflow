import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from functools import partial

# Set JAX to 64-bit precision for financial calculations
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=['lag_len', 'add_trend'])
def _adf_stat_single(y, lag_len, add_trend):
    """
    Calculates a single ADF statistic for a specific window y.
    Used within vmap for rolling calculations.
    """
    n = y.shape[0]
    
    # First differences
    dy = jnp.diff(y)
    
    # Target: dy[lag_len:]
    # Effective length for regression
    y_target = dy[lag_len:]
    eff_n = y_target.shape[0]
    
    # Regressors
    # 1. Constant
    const = jnp.ones(eff_n)
    
    # 2. Lagged Level: y[lag_len-1 : -1]
    lag_level = y[lag_len-1 : -1]
    
    # 3. Lagged Differences
    # We construct them dynamically
    # Since lag_len is static, we can unroll or stack
    lag_diffs = []
    for i in range(1, lag_len + 1):
        lag_diffs.append(dy[lag_len - i : -i])
    
    # Construct Design Matrix X
    if add_trend:
        # Canonical with Trend: Constant, Trend, Lagged Level, Lagged Diffs
        trend = jnp.arange(eff_n, dtype=jnp.float64)
        if lag_len > 0:
            X = jnp.column_stack([const, trend, lag_level] + lag_diffs)
        else:
            X = jnp.column_stack([const, trend, lag_level])
        # Index of Gamma (coefficient of lagged level) is 2 (0=const, 1=trend, 2=lag_level)
        gamma_idx = 2
    else:
        # Canonical without Trend: Constant, Lagged Level, Lagged Diffs
        if lag_len > 0:
            X = jnp.column_stack([const, lag_level] + lag_diffs)
        else:
            X = jnp.column_stack([const, lag_level])
        # Index of Gamma is 1 (0=const, 1=lag_level)
        gamma_idx = 1
    
    # OLS
    XtX = X.T @ X
    Xty = X.T @ y_target
    
    # Note: Ridge regularization removed to match PSY econometric specification.
    # Ensure min_length is sufficient to avoid singular matrices.
    
    beta = jnp.linalg.solve(XtX, Xty)
    
    # t-stat of gamma
    y_pred = X @ beta
    resid = y_target - y_pred
    ssr = jnp.sum(resid**2)
    var_resid = ssr / (eff_n - X.shape[1])
    
    XtX_inv = jnp.linalg.inv(XtX)
    se_gamma = jnp.sqrt(var_resid * XtX_inv[gamma_idx, gamma_idx])
    
    t_stat = beta[gamma_idx] / se_gamma
    return t_stat

@partial(jax.jit, static_argnames=['window', 'lags', 'add_trend'])
def _rolling_adf(prices, window, lags, add_trend):
    """
    Computes rolling ADF t-stats for a specific window size.
    """
    n = prices.shape[0]
    
    # Create windows (n - window + 1, window)
    starts = jnp.arange(n - window + 1)
    indices = starts[:, None] + jnp.arange(window)[None, :]
    windows = prices[indices]
    
    # Vmap the single ADF stat calculator
    calc_func = lambda w: _adf_stat_single(w, lags, add_trend)
    t_stats = jax.vmap(calc_func)(windows)
    
    return t_stats

def gsadf_values(series: pd.Series, min_length: int = None, add_trend: bool = False, lags: int = 1) -> pd.Series:
    """
    Calculates the GSADF (Generalized Supremum Augmented Dickey-Fuller) statistics.
    
    This implements the double recursion defined by Phillips, Shi, and Yu (2015):
    1. Inner Supremum (BSADF): For a fixed endpoint t, find max ADF over all start points.
    2. Outer Supremum (GSADF): For the sample up to t, find max BSADF.
    
    Args:
        series (pd.Series): Log prices.
        min_length (int): Minimum window size. If None, it is calculated 
                          using the PSY rule: r0 = 0.01 + 1.8/sqrt(T).
        add_trend (bool): If True, includes a linear time trend in the ADF specification.
                          Default False (Canonical: Constant, Lagged Level, Lagged Diffs).
        lags (int): Number of lags for ADF test.
        
    Returns:
        pd.Series: GSADF statistics over time.
    """
    arr = jnp.array(series.values, dtype=jnp.float64)
    n = len(series)
    
    # PSY Minimum Window Rule
    if min_length is None:
        r0 = 0.01 + 1.8 / np.sqrt(n)
        min_length = int(r0 * n)
        
    # Safety floor for min_length to ensure regression is solvable
    # Min regressors = 2 (const, level) + lags. If trend, +1.
    min_required = 5 + lags
    if min_length < min_required:
        min_length = min_required
        
    # Initialize BSADF array with a very small number
    # bsadf_arr[t] will store the max ADF statistic for windows ending at t
    bsadf_arr = np.full(n, -np.inf)
    
    # 1. Inner Recursion: Calculate BSADF sequence
    # We loop over all possible window lengths.
    for length in range(min_length, n + 1):
        # Get rolling ADF stats for this specific length
        stats = _rolling_adf(arr, length, lags, add_trend)
        stats_np = np.array(stats)
        
        # The stat at index i corresponds to the window ending at (i + length - 1)
        # So we align ends:
        end_indices = np.arange(length - 1, n)
        
        # Update maximums (BSADF logic: max over all start points for a fixed end t)
        current_vals = bsadf_arr[end_indices]
        bsadf_arr[end_indices] = np.maximum(current_vals, stats_np)

    # Replace initial -inf with NaN (before min_length)
    bsadf_arr[:min_length-1] = np.nan
    
    # 2. Outer Recursion: Calculate GSADF sequence
    # GSADF_t = max(BSADF_s) for s <= t
    # We use expanding max, carefully handling NaNs at the start
    
    # Fill leading NaNs with -inf for the accumulation step, then restore
    temp_bsadf = bsadf_arr.copy()
    temp_bsadf[np.isnan(temp_bsadf)] = -np.inf
    
    gsadf_arr = np.maximum.accumulate(temp_bsadf)
    
    # Restore NaNs where data wasn't sufficient
    gsadf_arr[:min_length-1] = np.nan
    
    return pd.Series(gsadf_arr, index=series.index, name="GSADF")