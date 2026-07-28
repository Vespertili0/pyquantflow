"""
 Labelling Module

This module provides implementations of the Symmetric CUSUM Filter as proposed
by Marcos Lopez de Prado. It includes functions for extracting CUSUM events
and JAX-accelerated functions for calibrating the optimal alpha threshold.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Union


def get_cusum_events(
    series: pd.Series,
    threshold: Union[float, pd.Series],
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM Filter as proposed by Marcos Lopez de Prado in
    Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.1.

    This is an event-driven filter designed to identify structural breaks/shocks
    in a series by monitoring cumulative deviations of its values.

    Parameters
    ----------
    series : pd.Series
        Series with a DatetimeIndex.
    threshold : float or pd.Series
        The threshold parameter (h). If a pd.Series is passed, it must align
        with the index of the series.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex of the filtered events.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series index must be a pandas DatetimeIndex.")

    # Align threshold if it is a Series
    if isinstance(threshold, pd.Series):
        # Reindex to series index and forward fill to handle any gaps
        threshold_aligned = threshold.reindex(series.index).ffill()
        threshold_arr = threshold_aligned.values
    else:
        threshold_arr = np.full(len(series), float(threshold))

    t_events = []
    s_pos = 0.0
    s_neg = 0.0

    series_arr = series.values
    times = series.index

    for i in range(len(series_arr)):
        val = series_arr[i]
        h = threshold_arr[i]

        # Ignore NaNs
        if np.isnan(val) or np.isnan(h):
            continue

        s_pos = max(0.0, s_pos + val)
        s_neg = min(0.0, s_neg + val)

        if s_pos >= h:
            s_pos = 0.0
            s_neg = 0.0
            t_events.append(times[i])
        elif s_neg <= -h:
            s_pos = 0.0
            s_neg = 0.0
            t_events.append(times[i])

    return pd.DatetimeIndex(t_events)


# ---------------------------------------------------------------------------
# JAX-accelerated CUSUM grid search helpers
# ---------------------------------------------------------------------------


def _cusum_scan_step(carry, x):
    """
    Single step function for ``jax.lax.scan`` implementing the symmetric
    CUSUM filter.

    The carry state ``(s_pos, s_neg)`` is frozen (not updated) when either
    the input value or threshold is NaN.  The triggered output is ``False``
    for NaN inputs, which is mathematically equivalent to the Python loop's
    ``continue`` statement.

    Parameters
    ----------
    carry : tuple[jax scalar, jax scalar]
        ``(s_pos, s_neg)`` — running positive and negative cumulative sums.
    x : tuple[jax scalar, jax scalar]
        ``(val, h_t)`` — current series value and threshold.

    Returns
    -------
    new_carry : tuple[jax scalar, jax scalar]
        Updated ``(s_pos, s_neg)`` after applying the CUSUM step.
    triggered : jax bool scalar
        ``True`` if the CUSUM threshold was crossed at this step.
    """
    s_pos, s_neg = carry
    val, h_t = x

    is_nan = jnp.isnan(val) | jnp.isnan(h_t)

    # Tentative updates (computed regardless of NaN to keep XLA graph simple)
    new_s_pos = jnp.maximum(0.0, s_pos + val)
    new_s_neg = jnp.minimum(0.0, s_neg + val)

    triggered_pos = new_s_pos >= h_t
    triggered_neg = new_s_neg <= -h_t
    triggered = triggered_pos | triggered_neg

    # Reset accumulators on trigger
    new_s_pos = jnp.where(triggered, 0.0, new_s_pos)
    new_s_neg = jnp.where(triggered, 0.0, new_s_neg)

    # Freeze carry on NaN — do not update state, do not fire event
    final_s_pos = jnp.where(is_nan, s_pos, new_s_pos)
    final_s_neg = jnp.where(is_nan, s_neg, new_s_neg)
    final_triggered = jnp.where(is_nan, False, triggered)

    return (final_s_pos, final_s_neg), final_triggered


def _run_cusum_for_alphas(
    series_arr: jnp.ndarray,
    vol_arr: jnp.ndarray,
    alphas_arr: jnp.ndarray,
) -> jnp.ndarray:
    """
    Vectorised CUSUM grid search over a 1-D array of alpha candidates.

    Uses ``jax.vmap`` to evaluate all alphas simultaneously and
    ``jax.lax.scan`` for the path-dependent CUSUM accumulation, avoiding
    the Python loop over the time axis.

    Parameters
    ----------
    series_arr : jnp.ndarray
        1-D array of shape ``(N,)`` — the input signal (e.g. returns).
    vol_arr : jnp.ndarray
        1-D array of shape ``(N,)`` — volatility series.
    alphas_arr : jnp.ndarray
        1-D array of shape ``(num_alphas,)`` — candidate multipliers.

    Returns
    -------
    jnp.ndarray
        Boolean array of shape ``(num_alphas, N)`` where ``True`` marks
        a triggered CUSUM event for that alpha at that time step.
    """

    def _scan_one_alpha(alpha):
        threshold = alpha * vol_arr
        xs = (series_arr, threshold)
        init_carry = (jnp.float32(0.0), jnp.float32(0.0))
        _, triggered = lax.scan(_cusum_scan_step, init_carry, xs)
        return triggered

    return jax.vmap(_scan_one_alpha)(alphas_arr)


# ---------------------------------------------------------------------------
# Public calibration API
# ---------------------------------------------------------------------------


def calibrate_cusum_alpha(
    series: pd.Series,
    target_events: Optional[int] = None,
    volatility: Optional[pd.Series] = None,
    alpha_min: float = 0.5,
    alpha_max: float = 3.0,
    alpha_step: float = 0.1,
    span: int = 100,
    objective: str = "budget",
    t1: Optional[pd.Series] = None,
) -> float:
    """
    Calibrates the CUSUM scalar multiplier ``alpha`` using one of two objectives.

    The grid search is accelerated by ``jax.vmap`` over the candidate alpha
    array and ``jax.lax.scan`` for the path-dependent accumulation, replacing
    the previous Python loop.  XLA compilation occurs on the first call per
    unique array shape; subsequent calls for the same series length execute
    in milliseconds.

    Parameters
    ----------
    series : pd.Series
        Series with a DatetimeIndex. Used for calibration (typically training
        fold only).
    target_events : Optional[int], default=None
        The target event count (budget). Required when ``objective="budget"``.
    volatility : Optional[pd.Series], default=None
        Pre-calculated volatility series. If None, dynamic EWMA volatility
        is calculated on the series.
    alpha_min : float, default=0.5
        Minimum multiplier to test.
    alpha_max : float, default=3.0
        Maximum multiplier to test.
    alpha_step : float, default=0.1
        Step size for the grid search.
    span : int, default=100
        EWMA span for volatility calculation when ``volatility`` is None.
    objective : str, default="budget"
        ``"budget"``    — select the alpha closest to ``target_events``.
        ``"uniqueness"`` — select the alpha that maximises the mean average
        sample uniqueness of the triggered events (requires ``t1``).
    t1 : Optional[pd.Series], default=None
        Barrier end-times series aligned to ``series.index``.
        Required when ``objective="uniqueness"``.

    Returns
    -------
    float
        The optimal alpha multiplier.

    Raises
    ------
    ValueError
        If ``objective="budget"`` and ``target_events`` is None.
        If ``objective="uniqueness"`` and ``t1`` is None.
        If ``objective`` is not one of ``"budget"`` or ``"uniqueness"``.
    """
    if objective == "budget" and target_events is None:
        raise ValueError("target_events is required when objective='budget'.")
    if objective == "uniqueness" and t1 is None:
        raise ValueError("t1 is required when objective='uniqueness'.")
    if objective not in ("budget", "uniqueness"):
        raise ValueError(
            f"Unknown objective '{objective}'. Must be 'budget' or 'uniqueness'."
        )

    if len(series) < 2:
        return alpha_min

    if volatility is None:
        vol = series.ewm(span=span).std()
    else:
        vol = volatility.reindex(series.index).ffill()

    # Generate alpha grid
    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2.0, alpha_step)

    # Convert to JAX arrays (float32 for XLA efficiency)
    series_jax = jnp.array(series.values, dtype=jnp.float32)
    vol_jax = jnp.array(vol.values, dtype=jnp.float32)
    alphas_jax = jnp.array(alphas, dtype=jnp.float32)

    # Vectorised CUSUM — shape: (num_alphas, N)
    event_mask_jax = _run_cusum_for_alphas(series_jax, vol_jax, alphas_jax)
    event_mask_jax.block_until_ready()
    event_mask = np.asarray(event_mask_jax, dtype=bool)

    if objective == "budget":
        event_counts = event_mask.sum(axis=1)  # shape: (num_alphas,)
        best_idx = int(np.argmin(np.abs(event_counts - target_events)))
        return float(alphas[best_idx])

    # objective == "uniqueness"
    from .sample_weights import get_sample_weights

    best_alpha = alpha_min
    best_score = -1.0

    for i, row_mask in enumerate(event_mask):
        event_times = series.index[row_mask]
        n_events = len(event_times)

        # Floor penalty: fewer than 10 events → score of 0 to prevent
        # the optimiser from starving the pipeline of data
        if n_events < 10:
            score = 0.0
        else:
            # Slice t1 to the CUSUM-selected events and compute uniqueness
            t1_slice = t1.reindex(event_times)
            # Drop any events without a valid barrier end-time
            t1_slice = t1_slice.dropna()

            if len(t1_slice) < 10:
                score = 0.0
            else:
                weights = get_sample_weights(t1_slice)
                score = float(weights.mean())

        if score > best_score:
            best_score = score
            best_alpha = alphas[i]

    return float(best_alpha)
