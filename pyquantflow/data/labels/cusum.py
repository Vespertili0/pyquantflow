import numpy as np
import pandas as pd


def get_cusum_events(
    prices: pd.Series,
    threshold: float | pd.Series,
) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM Filter as proposed by Marcos Lopez de Prado in
    Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.1.

    This is an event-driven filter designed to identify structural breaks/shocks
    in a price series by monitoring cumulative deviations of returns.

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex.
    threshold : float or pd.Series
        The threshold parameter (h). If a pd.Series is passed, it must align
        with the index of the daily returns (e.g. rolling daily volatility).

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex of the filtered events.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a pandas DatetimeIndex.")

    # Calculate returns (simple percent change)
    returns = prices.pct_change()

    # Align threshold if it is a Series
    if isinstance(threshold, pd.Series):
        # Reindex to returns index and forward fill to handle any gaps
        threshold_aligned = threshold.reindex(returns.index).ffill()
        threshold_arr = threshold_aligned.values
    else:
        threshold_arr = np.full(len(returns), float(threshold))

    t_events = []
    s_pos = 0.0
    s_neg = 0.0

    returns_arr = returns.values
    times = returns.index

    for i in range(len(returns_arr)):
        r = returns_arr[i]
        h = threshold_arr[i]

        # Ignore NaNs (e.g. the first entry of returns, or missing thresholds)
        if np.isnan(r) or np.isnan(h):
            continue

        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)

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
    prices: pd.Series,
    target_events: int,
    alpha_min: float = 0.5,
    alpha_max: float = 3.0,
    alpha_step: float = 0.1,
    span: int = 100,
) -> float:
    """
    Calibrates the CUSUM scalar multiplier alpha using the Event Budgeting technique.

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex. Used for calibration (typically training fold only).
    target_events : int
        The target event count (budget) for this series over the calibration period.
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
    if len(prices) < 2:
        return alpha_min

    returns = prices.pct_change()
    vol = returns.ewm(span=span).std()

    best_alpha = alpha_min
    min_diff = float("inf")

    # Generate alphas to sweep, avoiding floating point range issues
    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2.0, alpha_step)
    for alpha in alphas:
        threshold = alpha * vol
        events = get_cusum_events(prices, threshold)
        diff = abs(len(events) - target_events)

        if diff < min_diff:
            min_diff = diff
            best_alpha = alpha

    return float(best_alpha)
