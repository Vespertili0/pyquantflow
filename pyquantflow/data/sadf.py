import functools
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Enable 64-bit precision for JAX (crucial for matrix inversion stability in finance)
jax.config.update("jax_enable_x64", True)


def _get_y_x(
    series: pd.Series, model: str, lags: Union[int, list], add_const: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares the X and y datasets for SADF generation.
    (Kept largely in Pandas as this is a one-time setup step).
    """
    series = pd.DataFrame(series)
    series_diff = series.diff().dropna()
    x = _lag_df(series_diff, lags).dropna()
    x["y_lagged"] = series.shift(1).loc[x.index]  # add y_(t-1) column
    y = series_diff.loc[x.index]

    if add_const:
        x["const"] = 1

    # Add trend columns based on model type
    n_samples = x.shape[0]
    trend = np.arange(n_samples)
    
    if model == "linear":
        x["trend"] = trend
        beta_column = "y_lagged"
    elif model == "quadratic":
        x["trend"] = trend
        x["quad_trend"] = trend ** 2
        beta_column = "y_lagged"
    elif model == "sm_poly_1":
        y = series.loc[y.index]
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        x["trend"] = trend
        x["quad_trend"] = trend ** 2
        beta_column = "quad_trend"
    elif model == "sm_poly_2":
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        x["trend"] = trend
        x["quad_trend"] = trend ** 2
        beta_column = "quad_trend"
    elif model == "sm_exp":
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        x["trend"] = trend
        beta_column = "trend"
    elif model == "sm_power":
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        # Avoid log(0)
        with np.errstate(divide="ignore"):
            x["log_trend"] = np.log(trend)
            # Fix potential -inf at index 0 if trend starts at 0
            if trend[0] == 0: 
                 x.iloc[0, x.columns.get_loc("log_trend")] = 0
        beta_column = "log_trend"
    else:
        raise ValueError(f"Unknown model: {model}")

    # Move beta_column to the front (index 0) so JAX knows which coeff to pick
    columns = list(x.columns)
    if beta_column in columns:
        columns.insert(0, columns.pop(columns.index(beta_column)))
    x = x[columns]
    
    return x, y


def _lag_df(df: pd.DataFrame, lags: Union[int, list[int]]) -> pd.DataFrame:
    """Apply Lags to DataFrame"""
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + "_" + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how="outer")
    return df_lagged


@jax.jit
def _solve_ols_moment_stats(
    xx_window: jnp.ndarray, 
    xy_window: jnp.ndarray, 
    yy_window: jnp.ndarray, 
    n_obs: int
) -> float:
    """
    Solves OLS using precomputed moments matrices (X'X, X'y, y'y).
    Returns the t-statistic for the first coefficient (index 0).
    
    Math:
        beta = (X'X)^-1 X'y
        error = y - X*beta
        SSE = e'e = y'y - 2*beta'X'y + beta'(X'X)beta
        var(beta) = (SSE / (n - k)) * (X'X)^-1
    """
    k = xx_window.shape[0]
    
    # Solve for Beta: (X'X) * Beta = X'y
    # Using solve is numerically more stable than inv(xx) @ xy
    try:
        # Add a tiny jitter to diagonal for numerical stability if needed, 
        # though usually not needed with float64
        beta = jnp.linalg.solve(xx_window, xy_window)
        
        # We need the inverse of XX for the variance calculation
        xx_inv = jnp.linalg.inv(xx_window)
        
    except:
        # Fallback for singular matrices
        return jnp.nan

    # Calculate Sum of Squared Errors (SSE)
    # SSE = y'y - 2*beta'X'y + beta'X'X*beta
    # Note: xy_window is (k, 1), beta is (k, 1)
    term_1 = yy_window
    term_2 = 2 * (beta.T @ xy_window)
    term_3 = beta.T @ (xx_window @ beta)
    
    sse = term_1 - term_2 + term_3
    
    # Degrees of freedom: n - k
    # We use jnp.maximum to avoid division by zero or negative dof
    dof = jnp.maximum(n_obs - k, 1.0)
    
    mse = sse / dof
    
    # Variance of Beta
    # var(beta) = MSE * diag((X'X)^-1)
    beta_var = mse * jnp.diag(xx_inv).reshape(-1, 1)
    
    # t-stat = beta / sqrt(var(beta))
    # We are interested in index 0 (the moved beta_column)
    b_mean = beta[0, 0]
    b_std = jnp.sqrt(beta_var[0, 0])
    
    # Avoid division by zero
    t_stat = jnp.where(b_std > 1e-8, b_mean / b_std, jnp.nan)
    
    return t_stat


@functools.partial(jax.jit, static_argnames=['min_length', 'use_abs_penalty'])
def _run_sadf_kernel(
    X: jnp.ndarray, 
    y: jnp.ndarray, 
    min_length: int, 
    phi: float, 
    use_abs_penalty: bool
) -> jnp.ndarray:
    """
    Core JAX kernel for SADF.
    Uses Prefix Sums (CumSum) to perform OLS in constant time per window.
    """
    n_samples, n_features = X.shape
    
    # 1. Precompute Moments via Cumulative Sums
    # Pad with 0 at the beginning to handle subtraction easily
    # XX_t = sum(x_i * x_i^T) from 0 to t
    
    # Outer product for every row: (N, K, K)
    xx_moments = jax.vmap(lambda x_row: jnp.outer(x_row, x_row))(X)
    # Cross product for every row: (N, K, 1)
    xy_moments = jax.vmap(lambda x_row, y_row: jnp.outer(x_row, y_row))(X, y)
    # Squared y for every row: (N, 1, 1)
    yy_moments = y.reshape(-1, 1, 1) ** 2
    
    # Prefix sums (Cumulative Moments)
    # Shape becomes (N+1, ...), index i represents sum up to i (exclusive of i in standard python slice, but here represents inclusive of i-1)
    # We simply pad with zero at index 0.
    xx_cum = jnp.concatenate([jnp.zeros((1, n_features, n_features)), jnp.cumsum(xx_moments, axis=0)], axis=0)
    xy_cum = jnp.concatenate([jnp.zeros((1, n_features, 1)), jnp.cumsum(xy_moments, axis=0)], axis=0)
    yy_cum = jnp.concatenate([jnp.zeros((1, 1, 1)), jnp.cumsum(yy_moments, axis=0)], axis=0)

    # 2. Define the Scanning Function (Iterates over Time t)
    def scan_body(carry, t):
        # We are at time index `t`. (Note: t is 0-indexed relative to original data)
        # However, because of CumSum padding, `cum` array index `t+1` corresponds to data up to `t`.
        # The window ends at `t` (inclusive).
        
        end_idx_cum = t + 1
        
        # Identify valid start points.
        # Window length must be >= min_length.
        # Length = (t - start + 1) >= min_length  => start <= t + 1 - min_length
        # So valid starts are 0, 1, ..., t + 1 - min_length - 1
        
        max_start_idx = t - min_length + 1
        
        # We need to map over ALL possible starts up to N to keep shapes static for JIT.
        # We will mask out invalid results later.
        all_starts = jnp.arange(n_samples) 
        
        # Calculate Moments for window [start : end]
        # Moment[start:end] = CumSum[end] - CumSum[start]
        # In cum arrays: index `end_idx_cum` is sum(0..t). Index `start` is sum(0..start-1).
        
        xx_windows = xx_cum[end_idx_cum] - xx_cum[all_starts]
        xy_windows = xy_cum[end_idx_cum] - xy_cum[all_starts]
        yy_windows = yy_cum[end_idx_cum] - yy_cum[all_starts]
        
        lengths = (t + 1) - all_starts
        
        # Vectorized OLS solver over all start points
        # vmap over xx, xy, yy, lengths
        t_stats = jax.vmap(_solve_ols_moment_stats)(xx_windows, xy_windows, yy_windows, lengths)
        
        # Apply penalties and masking
        # 1. Mask invalid starts (where length < min_length or start > t)
        valid_mask = (lengths >= min_length) & (lengths > 0)
        
        # 2. Apply Phi penalty if needed
        # penalty = length ^ phi
        # If use_abs_penalty (SMT logic), take abs of t_stat
        
        stats_processed = jnp.where(use_abs_penalty, jnp.abs(t_stats), t_stats)
        
        if phi > 0.0:
            penalty = lengths ** phi
            stats_processed = stats_processed / penalty
        
        # Filter invalid windows
        # Set invalid stats to -inf so they don't affect the max
        stats_masked = jnp.where(valid_mask, stats_processed, -jnp.inf)
        
        # Take the Supremum (Max) over all valid start points for this t
        bsadf_t = jnp.max(stats_masked)
        
        # If all masked (e.g. t < min_length), return NaN
        bsadf_t = jnp.where(jnp.isneginf(bsadf_t), jnp.nan, bsadf_t)
        
        return carry, bsadf_t

    # 3. Run Scan over t
    time_indices = jnp.arange(n_samples)
    _, sadf_series_values = jax.lax.scan(scan_body, None, time_indices)
    
    return sadf_series_values


def get_sadf_jax(
    series: pd.Series,
    model: str,
    lags: Union[int, list],
    min_length: int,
    add_const: bool = False,
    phi: float = 0,
    verbose: bool = True, # verbose argument kept for compatibility, unused in JAX
) -> pd.Series:
    """
    JAX-Accelerated implementation of Supremum Augmented Dickey-Fuller (SADF).
    
    This function utilizes JAX for GPU/TPU acceleration and vectorized 
    linear algebra operations. It transforms the nested loop OLS structure 
    into a prefix-sum (cumulative moment) calculation, reducing algorithmic 
    complexity and enabling massive parallelism.

    Parameters
    ----------
    series : pd.Series
        Series for which SADF statistics are generated.
    model : str
        Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'.
    lags : int or list
        Either number of lags to use or array of specified lags.
    min_length : int
        Minimum number of observations needed for estimation.
    add_const : bool
        Flag to add constant.
    phi : float
        Coefficient to penalize large sample lengths when computing SMT, in [0, 1].
    verbose : bool
        Kept for API compatibility; JAX compilation logs may appear.

    Returns
    -------
    pd.Series
        SADF statistics indexed by the original series index (aligned to end points).
    """
    # 1. Prepare Data (Pandas/Numpy)
    X_df, y_df = _get_y_x(series, model, lags, add_const)
    
    # 2. Convert to JAX Arrays
    X_jax = jnp.array(X_df.values, dtype=jnp.float64)
    y_jax = jnp.array(y_df.values.reshape(-1, 1), dtype=jnp.float64)
    
    # 3. Determine specific logic flags
    # Original logic: if model[:2] == "sm", we use abs(adf) / length^phi
    use_abs_penalty = model.startswith("sm")
    
    # 4. Run JAX Kernel
    # The Kernel returns an array of size N (same as X rows).
    sadf_values = _run_sadf_kernel(
        X_jax, 
        y_jax, 
        min_length=min_length, 
        phi=phi, 
        use_abs_penalty=use_abs_penalty
    )
    
    # 5. Convert back to Pandas
    # The JAX output aligns with the X rows.
    # We should slice the output to match the valid range if necessary, 
    # but the original code returns a series indexed by molecule.
    # The JAX kernel computes for all t. Indices < min_length will be NaN.
    
    sadf_series = pd.Series(np.array(sadf_values), index=y_df.index)
    
    # Filter to valid min_length start (compatible with original behavior)
    sadf_series = sadf_series.iloc[min_length:]
    
    return sadf_series
