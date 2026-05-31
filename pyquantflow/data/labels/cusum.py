import numpy as np
import pandas as pd
from typing import Optional


def get_cusum_events(
    series: pd.Series,
    threshold: float | pd.Series,
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


def calibrate_cusum_alpha(
    series: pd.Series,
    target_events: int,
    volatility: Optional[pd.Series] = None,
    alpha_min: float = 0.5,
    alpha_max: float = 3.0,
    alpha_step: float = 0.1,
    span: int = 100,
) -> float:
    """
    Calibrates the CUSUM scalar multiplier alpha using the Event Budgeting technique.

    Parameters
    ----------
    series : pd.Series
        Series with a DatetimeIndex. Used for calibration (typically training fold only).
    target_events : int
        The target event count (budget) for this series over the calibration period.
    volatility : Optional[pd.Series], default=None
        Pre-calculated volatility series. If None, dynamic EWMA volatility will be
        calculated on the series.
    alpha_min : float, default=0.5
        Minimum multiplier to test.
    alpha_max : float, default=3.0
        Maximum multiplier to test.
    alpha_step : float, default=0.1
        Step size for grid search.
    span : int, default=100
        EWMA span for volatility calculation.

    Returns
    -------
    float
        The optimal alpha multiplier that brings the number of CUSUM events
        closest to the target_events budget.
    """
    if len(series) < 2:
        return alpha_min

    if volatility is None:
        vol = series.ewm(span=span).std()
    else:
        vol = volatility.reindex(series.index).ffill()

    best_alpha = alpha_min
    min_diff = float("inf")

    # Generate alphas to sweep, avoiding floating point range issues
    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2.0, alpha_step)
    for alpha in alphas:
        threshold = alpha * vol
        events = get_cusum_events(series, threshold)
        diff = abs(len(events) - target_events)

        if diff < min_diff:
            min_diff = diff
            best_alpha = alpha

    return float(best_alpha)
