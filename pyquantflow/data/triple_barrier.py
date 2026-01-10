import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from functools import partial

# Set JAX to 64-bit precision for financial calculations
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=['vertical_barrier'])
def _triple_barrier_jax(prices, volatility, vertical_barrier, pt, sl):
    """
    Vectorized Triple Barrier Method.
    
    Args:
        prices: array of prices.
        volatility: array of daily volatility (for dynamic thresholds).
        vertical_barrier: int, max steps to hold.
        pt: float, profit taking multiplier.
        sl: float, stop loss multiplier.
    
    Returns:
        labels: 1 (profit), -1 (loss), 0 (vertical barrier/time limit).
    """
    n = prices.shape[0]
    
    # We want to check for each t, what happens in [t+1, t+vertical_barrier]
    # We can construct a windowed view.
    
    # Create windows: (n, vertical_barrier)
    # Pad prices with nan or repeat last value to handle edge
    padded = jnp.pad(prices, (0, vertical_barrier), mode='edge')
    
    # Create indices
    idx = jnp.arange(n)[:, None] + jnp.arange(1, vertical_barrier + 1)[None, :]
    windows = padded[idx] # (n, vertical_barrier)
    
    # Calculate returns relative to price at t (prices[:, None])
    start_prices = prices[:, None]
    path_returns = (windows / start_prices) - 1.0
    
    # Dynamic barriers
    # upper = volatility * pt
    # lower = -volatility * sl
    upper = (volatility * pt)[:, None]
    lower = -(volatility * sl)[:, None]
    
    # Check breaches
    # We need the *first* breach.
    
    # 1. Profit Hits
    hit_upper = path_returns >= upper
    # 2. Loss Hits
    hit_lower = path_returns <= lower
    
    # Find first index of hit. argmax returns 0 if false, so we need to be careful.
    # We add a column of True at the end to ensure there is always a hit (the vertical barrier)
    # But actually, if neither hit, it's a 0 label.
    
    # We use a masking trick.
    # Create a range mask [1, 2, ... T]
    step_indices = jnp.arange(1, vertical_barrier + 1)
    
    # Upper hits indices (set non-hits to infinity)
    up_idx = jnp.where(hit_upper, step_indices, vertical_barrier + 9999)
    first_up = jnp.min(up_idx, axis=1)
    
    # Lower hits indices
    lo_idx = jnp.where(hit_lower, step_indices, vertical_barrier + 9999)
    first_lo = jnp.min(lo_idx, axis=1)
    
    # Determine Label
    # If first_up < first_lo and first_up <= vertical_barrier -> 1
    # If first_lo < first_up and first_lo <= vertical_barrier -> -1
    # Else -> 0
    
    label = jnp.zeros(n)
    label = jnp.where((first_up < first_lo) & (first_up <= vertical_barrier), 1.0, label)
    label = jnp.where((first_lo < first_up) & (first_lo <= vertical_barrier), -1.0, label)
    
    return label

def triple_barrier_labels(
    price_series: pd.Series, 
    volatility: pd.Series = None, 
    vertical_barrier_steps: int = 10, 
    pt: float = 1.0, 
    sl: float = 1.0
) -> pd.Series:
    """
    Generates labels (1, -1, 0) based on the Triple Barrier Method.
    1: Profit Take hit first.
    -1: Stop Loss hit first.
    0: Vertical Barrier (time limit) hit first.
    
    Args:
        price_series: Close prices.
        volatility: Volatility series (e.g., from an EWMA), aligned with prices.
                    If provided, pt and sl are multipliers of this volatility (dynamic barriers).
                    If None, pt and sl are treated as fixed percentage returns (e.g., 0.01 for 1%).
        vertical_barrier_steps: Number of steps (bars) for the time barrier.
        pt: Profit Taking multiplier (if vol provided) or fixed percentage (if vol is None).
        sl: Stop Loss multiplier (if vol provided) or fixed percentage (if vol is None).
    """
    # 1. Align Data
    if volatility is None:
        # Fixed barriers: Treat volatility as 1.0 everywhere so pt/sl become absolute thresholds
        common_idx = price_series.index
        p = jnp.array(price_series.values, dtype=jnp.float64)
        v = jnp.ones(len(p), dtype=jnp.float64)
    else:
        # Dynamic barriers: Align volatility and prices
        common_idx = price_series.index.intersection(volatility.index)
        p = jnp.array(price_series.loc[common_idx].values, dtype=jnp.float64)
        v = jnp.array(volatility.loc[common_idx].values, dtype=jnp.float64)
    
    # 2. Run JAX
    labels_jax = _triple_barrier_jax(p, v, vertical_barrier_steps, pt, sl)
    
    # 3. Sandwich
    # The last few labels might be biased due to padding in the JAX func, 
    # but the logic holds (they just hit the "edge" padding).
    # Usually we drop the last `vertical_barrier_steps` as we can't look forward.
    
    res = pd.Series(np.array(labels_jax), index=common_idx, name='bin')
    
    # Mask the end where look-forward wasn't fully possible
    res.iloc[-vertical_barrier_steps:] = np.nan
    
    return res