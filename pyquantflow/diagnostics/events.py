"""
Event Diagnostics Module

Provides visualisation wrappers for auditing CUSUM event trigger timestamps on single-asset
and multi-asset panels.
"""

import warnings
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
from typing import Dict, List, Optional
from ._renderer import DiagnosticResult, FigureFactory, DiagnosticWarning, PALETTE


def plot_cusum_events(
    df: pd.DataFrame,
    price_col: str = "Close",
    events: Optional[pd.Series] = None,
    ticker: str = "unknown",
) -> DiagnosticResult:
    """
    Renders price series (OHLC Candlestick or Line) with overlaid CUSUM event markers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data. Uses OHLC columns if present, otherwise falls back to `price_col`.
    price_col : str, default="Close"
        Name of the price column to render if OHLC columns are missing.
    events : Optional[pd.Series], default=None
        Series or DatetimeIndex of CUSUM event trigger timestamps.
    ticker : str, default="unknown"
        Symbol/name of the ticker for metadata logging.

    Returns
    -------
    DiagnosticResult
        Contains the Plotly figure and metadata (`n_events`, `event_density`, `ticker`).
    """
    fig = FigureFactory.create()

    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        df = df[~df.index.duplicated(keep="first")].sort_index()

    has_ohlc = all(col in df.columns for col in ["Open", "High", "Low", "Close"])

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[price_col], mode="lines", name="Price")
        )

    n_events = 0
    event_density = 0.0

    if events is not None and len(events) > 0:
        n_events = len(events)
        if len(df) > 1:
            days = (df.index[-1] - df.index[0]).days
            if days > 0:
                event_density = n_events / (days / 252)

        if isinstance(events, pd.DatetimeIndex):
            events_idx = events
        else:
            if isinstance(events, pd.Series) and pd.api.types.is_datetime64_any_dtype(
                events
            ):
                events_idx = pd.DatetimeIndex(events.values)
            elif isinstance(events, pd.Series):
                events_idx = pd.DatetimeIndex(events.index)
            else:
                events_idx = pd.DatetimeIndex(events)

        if df.index.tz is None:
            if getattr(events_idx, "tz", None) is not None:
                events_idx = events_idx.tz_convert(None)
        else:
            if getattr(events_idx, "tz", None) is None:
                events_idx = events_idx.tz_localize(df.index.tz)
            elif events_idx.tz != df.index.tz:
                events_idx = events_idx.tz_convert(df.index.tz)

        y_vals = df[price_col].reindex(events_idx, method="nearest")

        fig.add_trace(
            go.Scatter(
                x=events_idx,
                y=y_vals.values,
                mode="markers",
                marker_color=PALETTE["accent_1"],
                marker_symbol="line-ns-open",
                marker_size=12,
                name="CUSUM Events",
            )
        )

    metadata = {
        "n_events": n_events,
        "event_density": float(event_density),
        "ticker": ticker,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)


def plot_multi_asset_events(
    multi_asset_df: pd.DataFrame,
    tickers: List[str],
    max_tickers: int = 5,
    events_map: Optional[Dict[str, pd.DatetimeIndex]] = None,
) -> DiagnosticResult:
    """
    Renders stacked, synchronised price subplots across multiple tickers with CUSUM event overlays.

    Parameters
    ----------
    multi_asset_df : pd.DataFrame
        MultiIndex DataFrame with levels ("datetime", "ticker").
    tickers : List[str]
        List of ticker symbols to visualize.
    max_tickers : int, default=5
        Maximum number of subplots to render. Truncates and raises `DiagnosticWarning` if exceeded.
    events_map : Optional[Dict[str, pd.DatetimeIndex]], default=None
        Mapping of ticker symbols to CUSUM event timestamps.

    Returns
    -------
    DiagnosticResult
        Contains the multi-subplot Plotly figure and metadata (`n_events`, `event_density`, `tickers`).
    """
    if len(tickers) > max_tickers:
        warnings.warn(
            f"Number of tickers ({len(tickers)}) exceeds max_tickers ({max_tickers}). Truncating.",
            DiagnosticWarning,
            stacklevel=2,
        )
        tickers = tickers[:max_tickers]

    fig = plotly.subplots.make_subplots(
        rows=len(tickers),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=tickers,
    )

    total_events = 0
    total_trading_days = 0.0

    for i, ticker in enumerate(tickers):
        row = i + 1
        df_tk = multi_asset_df.xs(ticker, level="ticker")
        if not df_tk.index.is_unique or not df_tk.index.is_monotonic_increasing:
            df_tk = df_tk[~df_tk.index.duplicated(keep="first")].sort_index()

        has_ohlc = all(col in df_tk.columns for col in ["Open", "High", "Low", "Close"])
        if has_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=df_tk.index,
                    open=df_tk["Open"],
                    high=df_tk["High"],
                    low=df_tk["Low"],
                    close=df_tk["Close"],
                    name=f"{ticker} Price",
                ),
                row=row,
                col=1,
            )
        else:
            price_col = "Close" if "Close" in df_tk.columns else df_tk.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=df_tk.index,
                    y=df_tk[price_col],
                    mode="lines",
                    name=f"{ticker} Price",
                ),
                row=row,
                col=1,
            )

        if len(df_tk) > 1:
            total_trading_days += (df_tk.index[-1] - df_tk.index[0]).days / 252.0

        if events_map is not None and ticker in events_map:
            events = events_map[ticker]
            if events is not None and len(events) > 0:
                n_events = len(events)
                total_events += n_events

                price_col = "Close" if "Close" in df_tk.columns else df_tk.columns[0]

                if isinstance(events, pd.DatetimeIndex):
                    events_idx = events
                else:
                    if isinstance(
                        events, pd.Series
                    ) and pd.api.types.is_datetime64_any_dtype(events):
                        events_idx = pd.DatetimeIndex(events.values)
                    elif isinstance(events, pd.Series):
                        events_idx = pd.DatetimeIndex(events.index)
                    else:
                        events_idx = pd.DatetimeIndex(events)

                if df_tk.index.tz is None:
                    if getattr(events_idx, "tz", None) is not None:
                        events_idx = events_idx.tz_convert(None)
                else:
                    if getattr(events_idx, "tz", None) is None:
                        events_idx = events_idx.tz_localize(df_tk.index.tz)
                    elif events_idx.tz != df_tk.index.tz:
                        events_idx = events_idx.tz_convert(df_tk.index.tz)

                y_vals = df_tk[price_col].reindex(events_idx, method="nearest")

            fig.add_trace(
                go.Scatter(
                    x=events_idx,
                    y=y_vals.values,
                    mode="markers",
                    marker_color=PALETTE["accent_1"],
                    marker_symbol="line-ns-open",
                    marker_size=12,
                    name=f"{ticker} Events",
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        colorway=list(PALETTE.values()),
    )
    fig.update_xaxes(rangeslider_visible=False)

    metadata = {
        "n_events": total_events,
        "event_density": float(total_events / total_trading_days)
        if total_trading_days > 0
        else 0.0,
        "tickers": tickers,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
