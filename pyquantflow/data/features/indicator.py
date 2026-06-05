import numpy as np
import pandas as pd


def ICHIMOKU(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray = None,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> tuple:
    """
    Computes Ichimoku Cloud elements in a TA-Lib style.

    Args:
        high (np.ndarray or pd.Series): High prices.
        low (np.ndarray or pd.Series): Low prices.
        close (np.ndarray or pd.Series, optional): Close prices (needed for Chikou
            Span).
        tenkan_period (int): Period for Conversion Line (default 9).
        kijun_period (int): Period for Base Line (default 26).
        senkou_b_period (int): Period for Leading Span B (default 52).
        displacement (int): Displacement for Spans/Chikou (default 26).

    Returns:
        tuple: A tuple containing the following numpy arrays:
            (
                tenkan_sen,      # Conversion Line
                kijun_sen,       # Base Line
                span_a,          # Leading Span A (Projected recorded at current time)
                span_b,          # Leading Span B (Projected recorded at current time)
                span_a_shifted,  # Leading Span A (Shifted forward to align)
                span_b_shifted,  # Leading Span B (Shifted forward to align)
                chikou_span      # Lagging Span (Close shifted backwards) - None
            )
    """

    # Convert to Pandas Series for efficient rolling window calculations
    # This handles both numpy array and pandas series inputs gracefully
    high_s = pd.Series(high)
    low_s = pd.Series(low)

    # --- 1. Tenkan-sen (Conversion Line) ---
    tenkan_high = high_s.rolling(window=tenkan_period).max()
    tenkan_low = low_s.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # --- 2. Kijun-sen (Base Line) ---
    kijun_high = high_s.rolling(window=kijun_period).max()
    kijun_low = low_s.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2

    # --- 3. Senkou Span A (Leading Span A) ---
    # Recorded at current time t (Projected)
    span_a = (tenkan_sen + kijun_sen) / 2

    # --- 4. Senkou Span B (Leading Span B) ---
    # Recorded at current time t (Projected)
    span_b_high = high_s.rolling(window=senkou_b_period).max()
    span_b_low = low_s.rolling(window=senkou_b_period).min()
    span_b = (span_b_high + span_b_low) / 2

    # --- 5. Shifted Spans (The "Current Cloud") ---
    # Shifted forward to align with current price candle
    span_a_shifted = span_a.shift(displacement)
    span_b_shifted = span_b.shift(displacement)

    # --- 6. Chikou Span (Lagging Span) ---
    chikou_span = None
    #    if close is not None:
    #        close_s = pd.Series(close)
    #        chikou_span = close_s.shift(-displacement).to_numpy()

    # Return tuple of numpy arrays (TA-Lib style)
    return (
        tenkan_sen.to_numpy(),
        kijun_sen.to_numpy(),
        span_a.to_numpy(),
        span_b.to_numpy(),
        span_a_shifted.to_numpy(),
        span_b_shifted.to_numpy(),
        chikou_span,
    )


def ROGERSATCHELL(high, low, open, close, timeperiod=30):
    """
    Rogers-Satchell Volatility for TA-Lib style function calls.
    Uses the 'cumsum' trick to achieve near-C speeds without Numba or JAX.

    Parameters
    ----------
    high, low, open, close : np.ndarray
        Input price arrays (float). Must be the same length.
    timeperiod : int
        The rolling window size (default 30).

    Returns
    -------
    np.ndarray
        Volatility array of the same length as inputs.
        The first `timeperiod` elements are NaN.
    """
    # 1. Input Validation (TA-Lib style strictness)
    # Ensure inputs are float arrays to prevent integer division errors
    h_var = np.asarray(high, dtype=np.float64)
    l_var = np.asarray(low, dtype=np.float64)
    o_var = np.asarray(open, dtype=np.float64)
    c_var = np.asarray(close, dtype=np.float64)

    if not (h_var.shape == l_var.shape == o_var.shape == c_var.shape):
        raise ValueError("All input arrays must have the same shape.")

    n = h_var.shape[0]
    if timeperiod > n:
        # If data is shorter than window, return all NaNs
        return np.full(n, np.nan)

    # 2. Vectorised Math (Rogers-Satchell Formula)
    # rs = log(h/c)*log(h/o) + log(l/c)*log(l/o)
    # Use log(a/b) = log(a) - log(b) which is slightly safer/faster

    # term1 = ln(High / Close) * ln(High / Open)
    term1 = np.log(h_var / c_var) * np.log(h_var / o_var)

    # term2 = ln(Low / Close) * ln(Low / Open)
    term2 = np.log(l_var / c_var) * np.log(l_var / o_var)

    rs_daily = term1 + term2

    # 3. The Optimisation: Rolling Sum via Cumsum Trick
    # Sum[i:i+w] = CumSum[i+w] - CumSum[i]
    ret = np.cumsum(rs_daily, dtype=float)
    ret[timeperiod:] = ret[timeperiod:] - ret[:-timeperiod]

    # 4. Variance to Volatility
    # Divide by window size and sqrt
    vol = np.sqrt(ret / timeperiod)

    # 5. Padding
    # TA-Lib convention: if window is 30, indices 0..28 are NaN.
    # Index 29 is the first valid value.
    # Usually TA-Lib returns NaN for indices 0 to timeperiod-2.
    vol[: timeperiod - 1] = np.nan

    return vol


def EMA_RIBBON(
    open,
    high,
    low,
    close,
    timeperiods=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    rs_period=14,
):
    """
    Stationary Volatility-Scaled EMA Ribbon Engine.
    Requires IEEE 754 double-precision float arrays.
    Returns a tuple of 24 strictly stationary numpy arrays.
    """
    # 1. Input Integrity Check
    if np.any(open <= 0) or np.any(high <= 0) or np.any(low <= 0) or np.any(close <= 0):
        raise ValueError("Input asset prices must be strictly greater than 0.")

    T = len(close)
    M = len(timeperiods)
    epsilon = 1e-8

    # 2. Rogers-Satchell Volatility Estimator
    # \ln(H/C) \ln(H/O) + \ln(L/C) \ln(L/O)
    rs_term = np.log(high / close) * np.log(high / open) + np.log(low / close) * np.log(
        low / open
    )
    rs_mean = (
        pd.Series(rs_term, dtype=np.float64).rolling(window=rs_period).mean().values
    )

    # Enforce epsilon floor to prevent catastrophic downstream division-by-zero
    sigma_RS = np.maximum(np.sqrt(np.maximum(rs_mean, 0)), epsilon)

    # 3. Multi-Scale EMA Generation (Vectorized via Pandas C-backend)
    emas = np.empty((M, T), dtype=np.float64)
    for idx, k in enumerate(timeperiods):
        emas[idx, :] = pd.Series(close).ewm(span=k, adjust=False).mean().values

    # 4. Cross-Sectional Dimension Normalization (Micro-Spreads)
    # d_{i, t} = \ln(EMA_{k_{i+1}, t} / EMA_{k_i, t})
    d = np.log(emas[1:] / emas[:-1])
    d_tilde = d / sigma_RS  # Shape: (M-1, T)

    # 5. Macro-Spread
    # W_t = \ln(EMA_{k_M, t} / EMA_{k_1, t}) / \sigma_{RS, t}
    W_tilde = np.log(emas[-1] / emas[0]) / sigma_RS  # Shape: (T,)

    # 6. Cross-Sectional Shape Moments
    X = np.log(emas)  # Shape: (M, T)
    mu_x = np.mean(X, axis=0)
    sigma_x = np.std(X, axis=0)

    # Floor standard deviation to prevent skew/kurt division faults on flatlines
    sigma_x_safe = np.maximum(sigma_x, epsilon)

    skew_x = np.mean((X - mu_x) ** 3, axis=0) / (sigma_x_safe**3)
    kurt_x = np.mean((X - mu_x) ** 4, axis=0) / (sigma_x_safe**4)

    # 7. Structural Consensus Rank (Spearman Rank)
    # R_t calculates the ascending cross-sectional ranks across the M dimension
    R_t = np.argsort(np.argsort(X, axis=0), axis=0) + 1  # Shape: (M, T)
    rank_K = np.arange(1, M + 1, dtype=np.float64).reshape(M, 1)

    rank_diff_sq = (R_t - rank_K) ** 2
    sum_d2 = np.sum(rank_diff_sq, axis=0)

    # Standard Spearman formulation inverted (*) to map +1.0 to perfect bullish alignment
    # (where fast EMAs are larger than slow EMAs, driving a reversed mathematical rank)
    rho_t = -(1 - (6 * sum_d2) / (M * (M**2 - 1)))

    # 8. Kinematics (Ribbon Velocity)
    # \Delta \tilde{d}_{i, t} = \tilde{d}_{i, t} - \tilde{d}_{i, t-1}
    delta_d_tilde = np.zeros_like(d_tilde)
    delta_d_tilde[:, 1:] = d_tilde[:, 1:] - d_tilde[:, :-1]
    delta_d_tilde[:, 0] = np.nan

    # 9. Output Consolidation (24-Dimensional Matrix)
    outputs = [sigma_RS]  # Baseline Vol (1)

    for i in range(M - 1):
        outputs.append(d_tilde[i])  # Micro-Spreads (M-1)

    outputs.append(W_tilde)  # Macro-Spread (1)
    outputs.extend([sigma_x, skew_x, kurt_x, rho_t])  # Shape & Rank (4)

    for i in range(M - 1):
        outputs.append(delta_d_tilde[i])  # Kinematics (M-1)

    # 10. Cold-Start Warm-Up Masking
    W_start = max(timeperiods[-1], rs_period) * 3
    for arr in outputs:
        arr[:W_start] = np.nan

    return tuple(outputs)
