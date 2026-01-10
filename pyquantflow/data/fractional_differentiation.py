import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np


# Set JAX to 64-bit precision for financial calculations
jax.config.update("jax_enable_x64", True)


def _get_weights_ffd(d: float, thres: float, max_len: int):
    """
    Generates weights for the fractional differentiation.
    """
    k = jnp.arange(max_len)
    w = jnp.cumprod(jnp.concatenate([jnp.array([1.0]), (k[:-1] - d) / (k[:-1] + 1.0)]))
    
    # We only need weights where abs(w) > thres
    # But for JAX static shapes, we calculate all and mask later or cut manually.
    # To keep it JIT-friendly, we return all, and the convolution handles it.
    # However, to replicate standard behavior, we find the cutoff.
    return w

@jax.jit
def _apply_frac_diff(series, weights):
    """
    Applies the weights via convolution (JAX backend).
    """
    # Reverse weights for valid convolution alignment
    w_rev = weights[::-1]
    # 'valid' mode means we only compute where the window fully overlaps
    # But usually FFD returns a series of same length with NaNs at start.
    # We use 'full' and slice to emulate standard pandas behavior.
    res = jnp.convolve(series, w_rev, mode='full')
    return res[:series.shape[0]]

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
    arr = jnp.array(series.values, dtype=jnp.float64)
    
    # 2. Calculate Weights (Iterative logic done in numpy for dynamic sizing, then moved to JAX)
    # Note: Weight calc is fast enough in pure numpy/python usually.
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    weights = jnp.array(w)
    
    # 3. Apply via JAX
    res_jax = _apply_frac_diff(arr, weights)
    
    # 4. Sandwich back to Pandas
    # The first len(weights) - 1 elements are technically not valid because
    # they didn't have a full window. We assign NaN.
    result = np.array(res_jax)
    result[:len(weights) - 1] = np.nan
    
    return pd.Series(result, index=series.index, name=f"frac_diff_{d}")