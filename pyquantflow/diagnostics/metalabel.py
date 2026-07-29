"""
Meta-Labeling & Entropy Diagnostics

Validates whether the secondary meta-model is genuinely gating on primary 
prediction uncertainty (Shannon entropy) or filtering arbitrarily.
"""

import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, auc
from typing import Optional

from ._renderer import DiagnosticResult, FigureFactory, PALETTE


def plot_meta_label_entropy(
    df: pd.DataFrame,
    primary_pred_col: str = "primary_pred",
    entropy_col: str = "primary_entropy",
    target_col: str = "label",
    return_col: Optional[str] = None,
    decision_col: str = "final_decision",
) -> DiagnosticResult:
    """
    Renders the relationship between primary model entropy and secondary model filtering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictions, entropy, labels, and final decisions.
    primary_pred_col : str, default="primary_pred"
        Column name for primary model prediction.
    entropy_col : str, default="primary_entropy"
        Column name for primary model prediction entropy.
    target_col : str, default="label"
        Column name for true labels.
    return_col : Optional[str], default=None
        Column name for returns, used for y-axis if provided.
    decision_col : str, default="final_decision"
        Column name for secondary model binary decision.

    Returns
    -------
    DiagnosticResult
        Dual-panel figure with entropy scatter and decision histograms.
    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=("Primary Entropy vs True Target", "Entropy Distribution by Secondary Decision"),
    )

    y_col = return_col if return_col else target_col

    # Top panel: Scatter
    # We plot by label to color-encode
    label_colors = {
        2: PALETTE["tp"],       # Take Profit
        0: PALETTE["sl"],       # Stop Loss
        1: PALETTE["timeout"]   # Timeout
    }
    label_names = {
        2: "TP (2)",
        0: "SL (0)",
        1: "Timeout (1)"
    }

    # Ensure target_col is numeric for comparison if it's the default 0,1,2
    unique_labels = df[target_col].dropna().unique()
    
    for lbl in unique_labels:
        mask = df[target_col] == lbl
        color = label_colors.get(lbl, PALETTE["accent_1"])
        name = label_names.get(lbl, f"Label {lbl}")

        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, entropy_col],
                y=df.loc[mask, y_col],
                mode="markers",
                marker=dict(color=color, size=6, opacity=0.8),
                name=name,
                legendgroup="labels",
            ),
            row=1,
            col=1,
        )

    median_entropy_passed = float(df.loc[df[decision_col] == 1, entropy_col].median())
    if not np.isnan(median_entropy_passed):
        fig.add_vline(
            x=median_entropy_passed,
            row=1,
            col=1,
            line_dash="dash",
            line_color=PALETTE["accent_1"],
            annotation_text=f"Median H (passed)={median_entropy_passed:.3f}",
            annotation_position="top right",
        )

    # Bottom panel: Histograms
    fig.add_trace(
        go.Histogram(
            x=df.loc[df[decision_col] == 1, entropy_col],
            name="Passed (1)",
            marker_color=PALETTE["tp"],
            opacity=0.65,
            legendgroup="decisions",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=df.loc[df[decision_col] == 0, entropy_col],
            name="Filtered (0)",
            marker_color=PALETTE["sl"],
            opacity=0.65,
            legendgroup="decisions",
        ),
        row=2,
        col=1,
    )

    # Metadata computations
    # Spearman rho between H and return/label
    df_clean = df[[entropy_col, y_col]].dropna()
    if len(df_clean) > 2:
        entropy_return_spearman, _ = spearmanr(df_clean[entropy_col], df_clean[y_col])
    else:
        entropy_return_spearman = np.nan

    meta_filter_rate = float((df[decision_col] == 1).mean())
    median_entropy_filtered = float(df.loc[df[decision_col] == 0, entropy_col].median())

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        barmode="overlay",
        xaxis_title="Primary Entropy (H)",
        xaxis2_title="Primary Entropy (H)",
        yaxis_title=y_col,
        yaxis2_title="Count",
    )

    metadata = {
        "entropy_return_spearman": float(entropy_return_spearman),
        "meta_filter_rate": meta_filter_rate,
        "median_entropy_passed": median_entropy_passed,
        "median_entropy_filtered": median_entropy_filtered,
    }

    return DiagnosticResult(figure=fig, metadata=metadata)


def plot_meta_label_precision_recall(
    y_true: pd.Series,
    primary_preds: pd.Series,
    meta_preds: pd.Series,
    meta_probas: pd.Series,
) -> DiagnosticResult:
    """
    Plots Precision-Recall curves comparing primary and meta-filtered models.

    Parameters
    ----------
    y_true : pd.Series
        Binary true labels (1 for success, 0 for failure). Caller must binarize.
    primary_preds : pd.Series
        Primary model continuous predictions/scores.
    meta_preds : pd.Series
        Secondary model discrete predictions (unused for curve, kept for API matching).
    meta_probas : pd.Series
        Secondary model continuous probabilities.

    Returns
    -------
    DiagnosticResult
        Figure with PR curves and AUC-PR metadata.
    """
    fig = FigureFactory.create()

    # Primary Model PR Curve
    p_prec, p_rec, _ = precision_recall_curve(y_true, primary_preds)
    p_auc = auc(p_rec, p_prec)

    fig.add_trace(
        go.Scatter(
            x=p_rec,
            y=p_prec,
            mode="lines",
            name=f"Primary (AUC={p_auc:.3f})",
            line=dict(color=PALETTE["accent_1"], width=2),
        )
    )

    # Meta Model PR Curve
    m_prec, m_rec, _ = precision_recall_curve(y_true, meta_probas)
    m_auc = auc(m_rec, m_prec)

    fig.add_trace(
        go.Scatter(
            x=m_rec,
            y=m_prec,
            mode="lines",
            name=f"Meta-Filtered (AUC={m_auc:.3f})",
            line=dict(color=PALETTE["accent_2"], width=2),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        title="Precision-Recall Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
    )

    metadata = {
        "primary_auc_pr": float(p_auc),
        "meta_auc_pr": float(m_auc),
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
