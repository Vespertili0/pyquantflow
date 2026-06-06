import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union


def _adf_test_stat(series: pd.Series, lags: int = 1) -> float:
    """
    Computes the Augmented Dickey-Fuller t-statistic (constant only).
    Native implementation to avoid statsmodels dependency.
    """
    y = series.dropna().values
    if len(y) < lags + 2:
        return np.nan

    dy = np.diff(y)
    y_lag = y[:-1]
    Y = dy[lags:]
    n = len(Y)

    X = np.zeros((n, lags + 2))
    X[:, 0] = y_lag[lags - 1 : -1]  # y_{t-1}
    X[:, 1] = 1.0  # constant
    for i in range(lags):
        X[:, 2 + i] = dy[lags - 1 - i : -1 - i]  # lagged differences

    try:
        # Solve OLS: X * beta = Y
        beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan

    # Standard error of beta[0]
    # Check if residuals is empty (happens in exact fit)
    if len(residuals) == 0:
        residuals = np.sum((Y - X @ beta) ** 2)
    else:
        residuals = residuals[0]

    dof = max(1, n - X.shape[1])
    mse = residuals / dof

    try:
        cov_matrix = mse * np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(cov_matrix[0, 0])
    except np.linalg.LinAlgError:
        return np.nan

    if se_gamma == 0:
        return np.nan

    t_stat = beta[0] / se_gamma
    return float(t_stat)


def _adf_p_value(t_stat: float) -> float:
    """
    Approximates the MacKinnon p-value for the ADF test (constant only model)
    based on the t-statistic. Uses linear interpolation between key critical values.
    """
    if np.isnan(t_stat):
        return 1.0

    cv_1_pct = -3.43
    cv_5_pct = -2.86
    cv_10_pct = -2.57

    if t_stat <= cv_1_pct:
        return 0.01
    elif t_stat <= cv_5_pct:
        return 0.01 + 0.04 * (t_stat - cv_1_pct) / (cv_5_pct - cv_1_pct)
    elif t_stat <= cv_10_pct:
        return 0.05 + 0.05 * (t_stat - cv_5_pct) / (cv_10_pct - cv_5_pct)
    else:
        return 1.0


def adf_screened_ffd(
    series: Union[np.ndarray, pd.Series],
    d: Optional[float] = None,
    thres: float = 1e-4,
    significance_level: float = 0.05,
    d_grid: np.ndarray = np.arange(0.25, 1.05, 0.05),
) -> Tuple[pd.Series, float]:
    """
    ADF-screened Fixed-Width Window Fractional Differentiation.

    Operates in two modes:
        1. **Explicit mode** (d is not None): applies ``frac_diff_ffd`` directly
           using the given differencing order. No ADF screening is performed.
        2. **Screening mode** (d is None): iterates through ``d_grid`` to find
           the minimum d* that achieves stationarity (ADF p <= significance_level),
           preserving as much price memory as possible. Falls back to d=1.0 if
           no value in the grid achieves stationarity.

    Parameters
    ----------
    series : np.ndarray or pd.Series
        Input time series (e.g. raw prices or log prices).
    d : float or None, default None
        If provided, applies FFD with this fixed order (explicit mode).
        If None, runs the ADF screening loop (screening mode).
    thres : float, default 1e-4
        Weight cutoff threshold for FFD kernel truncation.
    significance_level : float, default 0.05
        Maximum ADF p-value to accept stationarity during screening.
    d_grid : np.ndarray, default np.arange(0.0, 1.05, 0.05)
        Grid of candidate differencing orders to evaluate during screening.

    Returns
    -------
    Tuple[pd.Series, float]
        A tuple of (differenced_series, d_used) where d_used is the
        differencing order that was applied.
    """
    # Coerce raw arrays to pd.Series
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    if d is not None:
        # Explicit mode: bypass screening entirely
        result = frac_diff_ffd(series, d=d, thres=thres)
        return result, d

    # Screening mode: find minimum d* achieving stationarity
    optimal_d = 1.0
    result = None

    for d_candidate in d_grid:
        diff_series = frac_diff_ffd(series, d=d_candidate, thres=thres)

        t_stat = _adf_test_stat(diff_series)
        p_value = _adf_p_value(t_stat)

        if p_value <= significance_level:
            optimal_d = d_candidate
            result = diff_series
            break

    # Fallback: if no d in the grid achieved stationarity, use d=1.0
    if result is None:
        result = frac_diff_ffd(series, d=optimal_d, thres=thres)

    return result, optimal_d


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
    arr = np.array(series.values, dtype=np.float64)

    # 2. Calculate Weights (Iterative logic done in numpy for dynamic sizing)
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    weights = np.array(w)

    # 3. Apply via pure NumPy convolution
    res = np.convolve(arr, weights[::-1], mode="full")

    # 4. Sandwich back to Pandas
    # The 'full' mode returns an array of size len(arr) + len(weights) - 1.
    # To match standard behavior where the index aligns with the end of the window:
    # First, discard the tail elements beyond the original series length.
    result = res[: len(arr)]

    # The first len(weights) - 1 elements are technically not valid because
    # they didn't have a full window. We assign NaN.
    result[: len(weights) - 1] = np.nan

    return pd.Series(result, index=series.index, name=f"frac_diff_{d}")
