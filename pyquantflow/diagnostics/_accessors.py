"""
AssetOrganiser Accessors Module

Side-effect module that binds diagnostic plotting wrapper methods directly onto `AssetOrganiser`.
"""

from pyquantflow.data.assetorganiser import AssetOrganiser
from .events import plot_multi_asset_events
from .uniqueness import plot_sample_concurrency
from ._renderer import DiagnosticResult


def _ao_plot_cusum_events(self) -> DiagnosticResult:
    """
    Convenience accessor: plots multi-asset CUSUM event overlays using cached CUSUM timestamps.

    Returns
    -------
    DiagnosticResult
        Multi-asset event diagnostic figure and metadata.

    Raises
    ------
    AttributeError
        If `downsample_to_cusum_events()` has not been called on the organiser.
    """
    if self.cusum_events_map is None:
        raise AttributeError(
            "cusum_events_map is None. Call downsample_to_cusum_events() first."
        )
    return plot_multi_asset_events(
        multi_asset_df=self.multi_asset,
        tickers=list(self.cusum_events_map.keys()),
        events_map=self.cusum_events_map,
    )


def _ao_plot_sample_concurrency(
    self, concurrency_threshold_pct: float = 0.75
) -> DiagnosticResult:
    """
    Convenience accessor: plots sample concurrency and uniqueness using organiser's `t1` and `weight` columns.

    Parameters
    ----------
    concurrency_threshold_pct : float, default=0.75
        Percentile threshold for shading concurrency warning bands.

    Returns
    -------
    DiagnosticResult
        Sample concurrency diagnostic figure and metadata.

    Raises
    ------
    KeyError
        If `apply_continuous_labels()` has not been called to generate the `t1` column.
    """
    if "t1" not in self.multi_asset.columns:
        raise KeyError("'t1' column missing. Call apply_continuous_labels() first.")
    weight_col = self.weight_col if self.weight_col else "weight"
    weight_series = (
        self.multi_asset[weight_col] if weight_col in self.multi_asset.columns else None
    )
    return plot_sample_concurrency(
        t1_series=self.multi_asset["t1"],
        weight_series=weight_series,
        concurrency_threshold_pct=concurrency_threshold_pct,
    )


AssetOrganiser.plot_cusum_events = _ao_plot_cusum_events
AssetOrganiser.plot_sample_concurrency = _ao_plot_sample_concurrency
