"""
CV Split Leakage Auditor

Provides visual proof that training label event horizons do not overlap with test
evaluation windows, and exposes fold-to-fold feature distribution drift.
"""

import warnings
import pandas as pd
import plotly.graph_objects as go
from typing import Union

from ._renderer import DiagnosticResult, FigureFactory, DiagnosticWarning, PALETTE
from pyquantflow.model.cross_validation import PurgedKFoldCV, CombinatorialPurgedKFold


def plot_cv_splits(
    cv_splitter: Union[PurgedKFoldCV, CombinatorialPurgedKFold],
    X: pd.DataFrame,
    y: pd.Series,
) -> DiagnosticResult:
    """
    Renders a Gantt-style timeline of CV splits and exposes t1 horizon leakage.

    Parameters
    ----------
    cv_splitter : PurgedKFoldCV or CombinatorialPurgedKFold
        The cross-validation splitter instance.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training targets.

    Returns
    -------
    DiagnosticResult
        Figure with Gantt chart of splits and leakage metadata.
    """
    fig = FigureFactory.create()

    # Generate splits
    splits = list(cv_splitter.split(X, y))
    n_splits = len(splits)

    # Extract datetime index
    if isinstance(X.index, pd.MultiIndex):
        times = X.index.get_level_values("datetime")
    else:
        times = X.index

    is_cpcv = isinstance(cv_splitter, CombinatorialPurgedKFold)

    # Get t1 series (if available)
    if is_cpcv:
        t1 = None
    else:
        # t1 could be a string or a pd.Series on PurgedKFoldCV
        if isinstance(cv_splitter.t1, pd.Series):
            t1 = cv_splitter.t1
        elif isinstance(cv_splitter.t1, str) and cv_splitter.t1 in X.columns:
            t1 = X[cv_splitter.t1]
        else:
            t1 = None

    has_leakage = False
    leaking_fold_indices = []
    
    is_datetime = pd.api.types.is_datetime64_any_dtype(times)

    for i, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train_start_ts = times[train_idx[0]]
        train_end_ts = times[train_idx[-1]]
        test_start_ts = times[test_idx[0]]
        test_end_ts = times[test_idx[-1]]

        if is_datetime:
            train_duration = (train_end_ts - train_start_ts).total_seconds() * 1000  # ms
            test_duration = (test_end_ts - test_start_ts).total_seconds() * 1000
        else:
            train_duration = train_end_ts - train_start_ts
            test_duration = test_end_ts - test_start_ts

        # Check leakage
        if t1 is not None:
            # We want to find if any t1 from train overlaps with the test window.
            # Assuming time series order, check if any t1 in train before test_start
            # has a horizon that falls within the test window.
            train_times = times[train_idx]
            train_t1s = t1.iloc[train_idx]

            # Identify train samples before test split
            mask_before = train_times < test_start_ts
            if mask_before.any():
                t1s_before = train_t1s[mask_before]
                # A leak occurs if a training sample's event end time extends into or past the test window
                leaks = (t1s_before >= test_start_ts)
                if leaks.any():
                    has_leakage = True
                    leaking_fold_indices.append(i)

        # Plot Train Segment
        # NOTE: Plotly Bar with datetime base and timedelta x has quirks.
        # Using ms for duration and datetime for base works well with go.Bar(orientation='h').
        fig.add_trace(
            go.Bar(
                base=train_start_ts,
                x=[train_duration],
                y=[f"Fold {i}"],
                orientation="h",
                marker_color="steelblue",
                name="Train",
                showlegend=(i == 0),
            )
        )

        # Plot Test Segment
        fig.add_trace(
            go.Bar(
                base=test_start_ts,
                x=[test_duration],
                y=[f"Fold {i}"],
                orientation="h",
                marker_color="darkorange",
                name="Test",
                showlegend=(i == 0),
            )
        )

        # Purged zone
        fig.add_vrect(
            x0=train_end_ts,
            x1=test_start_ts,
            fillcolor=PALETTE["sl"],
            opacity=0.15,
            layer="below",
            line_width=0,
        )

        # Embargo zone computation
        # Approximate visually by using test_end_ts + some delta.
        # This is purely visual representation for the Gantt.
        # A true embargo end time depends on unique times in dataset.
        if is_datetime:
            embargo_end_ts = test_end_ts + pd.Timedelta(days=5)  # Visual approximation
        else:
            embargo_end_ts = test_end_ts + max(5, int(test_duration * 0.1))
        fig.add_vrect(
            x0=test_end_ts,
            x1=embargo_end_ts,
            fillcolor="grey",
            opacity=0.10,
            layer="below",
            line_width=0,
        )

        # Plot t1 tick marks
        if t1 is not None:
            t1_train_vals = t1.iloc[train_idx].dropna().values
            fig.add_trace(
                go.Scatter(
                    x=t1_train_vals,
                    y=[f"Fold {i}"] * len(t1_train_vals),
                    mode="markers",
                    marker_symbol="line-ns-open",
                    marker_color=PALETTE["timeout"],
                    showlegend=(i == 0),
                    name="t₁ horizons",
                )
            )

    if has_leakage:
        warnings.warn(
            f"Leakage detected in folds: {leaking_fold_indices}. t1 events extend into test sets.",
            DiagnosticWarning,
            stacklevel=2,
        )

    layout_kwargs = dict(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        barmode="overlay",
        title="Cross-Validation Splits & Purging",
    )
    if is_datetime:
        layout_kwargs["xaxis_type"] = "date"
    else:
        layout_kwargs["xaxis_type"] = "linear"
        
    fig.update_layout(**layout_kwargs)

    metadata = {
        "n_splits": n_splits,
        "has_leakage": has_leakage,
        "leaking_fold_indices": leaking_fold_indices,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)


def plot_fold_feature_drift(
    X: pd.DataFrame,
    cv_splitter: Union[PurgedKFoldCV, CombinatorialPurgedKFold],
    feature_col: str,
) -> DiagnosticResult:
    """
    Renders distribution drift of a specific feature across CV fold training sets.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    cv_splitter : PurgedKFoldCV or CombinatorialPurgedKFold
        The cross-validation splitter instance.
    feature_col : str
        The feature column to plot.

    Returns
    -------
    DiagnosticResult
        Figure with violin plots per fold and feature drift metadata.
    """
    fig = FigureFactory.create()

    # We just need a dummy y to generate splits
    dummy_y = pd.Series(0, index=X.index)
    splits = list(cv_splitter.split(X, dummy_y))

    for i, (train_idx, _) in enumerate(splits):
        train_vals = X.iloc[train_idx][feature_col].dropna().values

        fig.add_trace(
            go.Violin(
                y=train_vals,
                name=f"Fold {i}",
                box_visible=True,
                meanline_visible=True,
                line_color=PALETTE["accent_1"],
            )
        )

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        title=f"Feature Drift: {feature_col} (Training Folds)",
    )

    metadata = {
        "n_splits": len(splits),
        "feature_col": feature_col,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
