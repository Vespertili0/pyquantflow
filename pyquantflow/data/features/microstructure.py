"""
Market Microstructure Feature Estimators
=========================================

Implements classical microstructure spread estimators adapted for standard
OHLCV bar data. All functions follow the same architectural contract as
``FRACTIONAL_DIFF`` and ``SADF_JAX`` in ``indicator.py``:

- Accept ``pd.Series`` or ``np.ndarray`` inputs.
- Perform fully vectorised computation via pandas rolling windows.
- Return a ``np.ndarray`` of **identical length** to the input, with
  cold-start indices padded with ``np.nan``.

References
----------
Roll, R. (1984). *A simple implicit measure of the effective bid-ask spread
in an efficient market*. The Journal of Finance, 39(4), 1127–1139.

Corwin, S. A., & Schultz, P. (2012). *A simple way to estimate bid-ask
spreads from daily high and low prices*. The Journal of Finance, 67(2),
719–760.
"""

import numpy as np
import pandas as pd


def _roll_single(series: pd.Series, window: int) -> np.ndarray:
    n = len(series)
    if n < window + 1:
        return np.full(n, np.nan)
    delta = series.diff()
    delta_lag = delta.shift(1)
    cov = delta.rolling(window=window).cov(delta_lag)
    spread = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    result = spread.to_numpy(dtype=np.float64, copy=True)
    result[: window + 1] = np.nan
    return result


def ROLL_MEASURE(
    close: np.ndarray | pd.Series,
    window: int = 20,
) -> np.ndarray:
    """
    Roll's (1984) Effective Bid-Ask Spread Estimator.

    Estimates the effective bid-ask spread from the serial covariance of
    price changes. The formula is:

    .. math::

        \\text{spread}_t = 2 \\sqrt{\\max(-\\text{Cov}(\\Delta P_t,
        \\Delta P_{t-1}),\\ 0)}

    where :math:`\\Delta P_t = P_t - P_{t-1}` is the price change.

    A negative serial covariance of returns is a hallmark of the bid-ask
    bounce effect: prices tend to revert after each trade due to the
    alternating execution between the bid and ask sides.

    Parameters
    ----------
    close : np.ndarray or pd.Series
        Raw close price series.
    window : int, default 20
        Rolling window length for the covariance calculation.
        Must be at least 2. The first ``window`` values are ``np.nan``.

    Returns
    -------
    np.ndarray
        Estimated bid-ask spread series. Same length as input.
        Cold-start (first ``window``) indices are ``np.nan``.
        Values that would produce a negative covariance (implying a
        positive autocorrelation, inconsistent with Roll's model) are
        clamped to ``0``.

    Notes
    -----
    This estimator is most reliable on liquid, high-frequency OHLCV data.
    On low-frequency daily bars, the spread will be compressed relative to
    the true intraday effective spread.

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([100.0, 100.5, 99.5, 100.5, 99.5, 100.5])
    >>> spreads = ROLL_MEASURE(close, window=3)
    """
    if window < 2:
        raise ValueError("'window' must be at least 2.")

    if isinstance(close, pd.Series) and isinstance(close.index, pd.MultiIndex):
        from pyquantflow.data.utils import safe_stack

        unstacked = close.unstack(level="ticker")
        transformed = unstacked.apply(
            lambda col: pd.Series(_roll_single(col, window=window), index=col.index)
        )
        stacked = safe_stack(transformed, level="ticker")
        if stacked.index.names != close.index.names:
            stacked = stacked.reorder_levels(close.index.names)
        return stacked.reindex(close.index).to_numpy(dtype=np.float64, copy=True)

    series = pd.Series(close) if isinstance(close, np.ndarray) else close.copy()
    return _roll_single(series, window=window)


def _corwin_schultz_single(
    high_s: pd.Series, low_s: pd.Series, window: int
) -> np.ndarray:
    n = len(high_s)
    if n < window + 2:
        return np.full(n, np.nan)

    if (high_s.dropna() <= 0).any() or (low_s.dropna() <= 0).any():
        raise ValueError("'high' and 'low' prices must be strictly positive.")

    log_hl_sq = np.log(high_s / low_s) ** 2
    two_day_sum = log_hl_sq + log_hl_sq.shift(1)

    composite_high = np.fmax(high_s, high_s.shift(1))
    composite_low = np.fmin(low_s, low_s.shift(1))
    gamma_raw = np.log(composite_high / composite_low) ** 2

    # Smooth both beta and gamma over the rolling window for consistent scale
    beta = two_day_sum.rolling(window=window).mean()
    gamma = gamma_raw.rolling(window=window).mean()

    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha_clamped = alpha.clip(lower=0.0)

    exp_alpha = np.exp(alpha_clamped)
    spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

    result = spread.to_numpy(dtype=np.float64, copy=True)
    result[: window + 1] = np.nan
    return result


def CORWIN_SCHULTZ(
    high: np.ndarray | pd.Series,
    low: np.ndarray | pd.Series,
    window: int = 20,
) -> np.ndarray:
    """
    Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator.

    Estimates the bid-ask spread from the variance of overlapping 2-day
    High/Low log-price ratios. The key insight is that the range of prices
    over two consecutive days increases with the square root of time, while
    the bid-ask component increases linearly — allowing the spread to be
    isolated.

    The derivation:

    .. math::

        \\beta &= \\mathbb{E}\\left[\\sum_{j=0}^{1}
            \\left(\\ln \\frac{H_{t-j}}{L_{t-j}}\\right)^2\\right]

        \\gamma &= \\left(\\ln \\frac{H_{t,t-1}^{\\max}}{
            L_{t,t-1}^{\\min}}\\right)^2

        \\alpha &= \\frac{\\sqrt{2\\beta} - \\sqrt{\\beta}}{3 - 2\\sqrt{2}}
            - \\sqrt{\\frac{\\gamma}{3 - 2\\sqrt{2}}}

        \\text{spread} &= \\frac{2(e^\\alpha - 1)}{1 + e^\\alpha}

    Parameters
    ----------
    high : np.ndarray or pd.Series
        Daily high price series.
    low : np.ndarray or pd.Series
        Daily low price series.
    window : int, default 20
        Rolling window for smoothing the :math:`\\beta` component.
        Must be at least 2. The first ``window + 1`` values are ``np.nan``.

    Returns
    -------
    np.ndarray
        Estimated bid-ask spread series as a fraction (e.g., 0.01 = 1%).
        Same length as input. Cold-start indices are ``np.nan``.
        Negative ``alpha`` values (theoretically invalid) are clamped to
        produce a spread of ``0``.

    Notes
    -----
    Both ``high`` and ``low`` must have the same length and be strictly
    positive. A ``ValueError`` is raised if lengths differ.

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([101.0, 102.0, 101.5, 103.0, 102.0, 103.5])
    >>> low  = np.array([ 99.0,  98.0,  99.5,  97.0,  98.0,  96.5])
    >>> spreads = CORWIN_SCHULTZ(high, low, window=3)
    """
    if window < 2:
        raise ValueError("'window' must be at least 2.")
    if len(high) != len(low):
        raise ValueError("'high' and 'low' must have the same length.")

    if (
        isinstance(high, pd.Series)
        and isinstance(low, pd.Series)
        and isinstance(high.index, pd.MultiIndex)
    ):
        from pyquantflow.data.utils import safe_stack

        h_unstacked = high.unstack(level="ticker")
        l_unstacked = low.unstack(level="ticker")
        res_cols = {}
        for tk in h_unstacked.columns:
            res_cols[tk] = pd.Series(
                _corwin_schultz_single(h_unstacked[tk], l_unstacked[tk], window=window),
                index=h_unstacked.index,
            )
        transformed = pd.DataFrame(res_cols, index=h_unstacked.index)
        transformed.columns.name = "ticker"
        stacked = safe_stack(transformed, level="ticker")
        if stacked.index.names != high.index.names:
            stacked = stacked.reorder_levels(high.index.names)
        return stacked.reindex(high.index).to_numpy(dtype=np.float64, copy=True)

    high_s = pd.Series(high) if isinstance(high, np.ndarray) else high.reset_index(drop=True)
    low_s = pd.Series(low) if isinstance(low, np.ndarray) else low.reset_index(drop=True)
    return _corwin_schultz_single(high_s, low_s, window=window)
