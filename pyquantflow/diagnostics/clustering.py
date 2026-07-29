"""
Clustering & Multicollinearity Diagnostics

Visualises feature cluster structures and Importance scores (MDA/SFI) side-by-side
to help resolve multicollinearity distortion.
"""

import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import Optional, Dict, Union

from ._renderer import DiagnosticResult, PALETTE


def plot_feature_clusters(
    regime_results: Union[Dict[int, Dict[str, pd.DataFrame]], pd.DataFrame],
    correlation_matrix: pd.DataFrame,
    linkage_matrix: Optional[np.ndarray] = None,
    method: str = "ward",
    regime_id: Optional[int] = None,
) -> DiagnosticResult:
    """
    Renders a dendrogram-ordered correlation heatmap alongside cluster importance scores.

    Parameters
    ----------
    regime_results : dict or pd.DataFrame
        Either the raw nested dict from `evaluate_importance()` or the
        consolidated `importance_df` from `FeatureEvaluator`.
    correlation_matrix : pd.DataFrame
        Pearson correlation matrix of the features.
    linkage_matrix : Optional[np.ndarray], default=None
        Pre-computed linkage matrix. If None, it is computed internally.
    method : str, default="ward"
        Linkage method to use if computing linkage internally.
    regime_id : Optional[int], default=None
        Specific regime ID to plot. If None, aggregates across all regimes.

    Returns
    -------
    DiagnosticResult
        Dual-panel figure with correlation heatmap and importance bar chart.
    """
    # 1. Prepare Data
    if isinstance(regime_results, pd.DataFrame):
        # We received the consolidated DataFrame
        if regime_id is not None:
            # Filter by regime_id (level 0)
            df = regime_results.loc[regime_id].copy()
            df = df.reset_index()
        else:
            # Aggregate across regimes
            df = (
                regime_results.groupby(level=1)
                .agg(
                    {
                        "features": "first",
                        "sfi_mean": "mean",
                        "sfi_std": "mean",
                        "mda_mean": "mean",
                        "mda_std": "mean",
                    }
                )
                .reset_index()
            )
    else:
        # We received the raw nested dict
        if regime_id is not None:
            if regime_id not in regime_results:
                raise KeyError(f"Regime {regime_id} not found in regime_results")

            sfi_df = regime_results[regime_id]["SFI"]
            mda_df = regime_results[regime_id]["MDA"]
            df = sfi_df.join(mda_df[["mda_mean", "mda_std"]], how="outer").reset_index()
        else:
            # Aggregate across regimes
            regime_dfs = []
            for r_id, inner_dict in regime_results.items():
                sfi_df = inner_dict["SFI"]
                mda_df = inner_dict["MDA"]
                merged = sfi_df.join(mda_df[["mda_mean", "mda_std"]], how="outer")
                regime_dfs.append(merged)

            df_all = pd.concat(regime_dfs, axis=0).reset_index()
            df = (
                df_all.groupby("cluster_id")
                .agg(
                    {
                        "features": "first",
                        "sfi_mean": "mean",
                        "sfi_std": "mean",
                        "mda_mean": "mean",
                        "mda_std": "mean",
                    }
                )
                .reset_index()
            )

    # Build cluster assignments map
    cluster_assignments = {}
    for _, row in df.iterrows():
        c_id = int(row["cluster_id"])
        features_str = str(row["features"])
        # Split features by comma and trim whitespace
        for feat in (f.strip() for f in features_str.split(",") if f.strip()):
            cluster_assignments[feat] = c_id

    # 2. Linkage Computation
    if linkage_matrix is None:
        dist = np.sqrt(0.5 * (1 - correlation_matrix.fillna(0).clip(-1, 1)))
        condensed = ssd.squareform(dist.values, checks=False)
        linkage_matrix = sch.linkage(condensed, method=method)

    # Get dendrogram leaf ordering
    dendro = sch.dendrogram(linkage_matrix, no_plot=True)
    leaves = dendro["leaves"]

    # Reorder correlation matrix
    ordered_labels = [correlation_matrix.columns[i] for i in leaves]
    reordered_corr = correlation_matrix.iloc[leaves, leaves]

    # 3. Build Figure
    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            "Feature Correlation (Dendrogram Sorted)",
            "Cluster Importance (MDA vs SFI)",
        ),
        horizontal_spacing=0.1,
    )

    # Left panel: Heatmap
    fig.add_trace(
        go.Heatmap(
            z=reordered_corr.values,
            x=ordered_labels,
            y=ordered_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Correlation", x=0.5, thickness=15),
        ),
        row=1,
        col=1,
    )

    # Right panel: Bar Chart
    PALETTE_LIST = list(PALETTE.values())
    sorted_cluster_ids = sorted(df["cluster_id"].unique())
    cluster_colour = {
        cid: PALETTE_LIST[i % len(PALETTE_LIST)]
        for i, cid in enumerate(sorted_cluster_ids)
    }

    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        c_colour = cluster_colour[cid]
        c_name = f"Cluster {cid}"

        # MDA Bar
        fig.add_trace(
            go.Bar(
                name=f"{c_name} MDA",
                x=[row["mda_mean"]],
                y=[c_name],
                error_x=dict(type="data", array=[row["mda_std"]], visible=True),
                marker_color=c_colour,
                orientation="h",
                legendgroup=c_name,
            ),
            row=1,
            col=2,
        )

        # SFI Bar
        fig.add_trace(
            go.Bar(
                name=f"{c_name} SFI",
                x=[row["sfi_mean"]],
                y=[c_name],
                error_x=dict(type="data", array=[row["sfi_std"]], visible=True),
                marker_color=c_colour,
                marker_pattern_shape="/",
                orientation="h",
                legendgroup=c_name,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        font={"family": "Inter, DM Mono, monospace", "size": 13},
        paper_bgcolor="#0F0F13",
        plot_bgcolor="#0F0F13",
        barmode="group",
        height=600,
    )

    # Find top cluster
    if len(df) > 0:
        top_idx = df["mda_mean"].idxmax()
        top_cluster_id = int(df.loc[top_idx, "cluster_id"])
        top_cluster_mda = float(df.loc[top_idx, "mda_mean"])
    else:
        top_cluster_id = -1
        top_cluster_mda = 0.0

    metadata = {
        "cluster_assignments": cluster_assignments,
        "top_cluster_id": top_cluster_id,
        "top_cluster_mda": top_cluster_mda,
        "n_clusters": len(sorted_cluster_ids),
        "n_features": len(correlation_matrix.columns),
    }

    return DiagnosticResult(figure=fig, metadata=metadata)
