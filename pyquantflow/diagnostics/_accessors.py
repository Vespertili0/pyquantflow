"""
AssetOrganiser Accessors Module

Side-effect module that binds diagnostic plotting wrapper methods directly onto `AssetOrganiser`.
"""

import pandas as pd
from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.data.sk_transformers import GSADFTransformer
from pyquantflow.model.classifier import PrimarySecondaryClassifier
from pyquantflow.model.cross_validation import (
    CombinatorialPurgedKFold,
    PurgedKFoldCV,
)
from pyquantflow.model.feature_evaluation import (
    FeatureEvaluator,
    StationaryTransformer,
)
from ._renderer import DiagnosticResult
from .events import plot_multi_asset_events
from .uniqueness import plot_sample_concurrency


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

    t1_series = self.multi_asset["t1"]
    if isinstance(t1_series.index, pd.MultiIndex):
        if "ticker" in t1_series.index.names:
            t1_series = t1_series.reset_index(level="ticker", drop=True)

    weight_col = self.weight_col if self.weight_col else "weight"
    weight_series = (
        self.multi_asset[weight_col] if weight_col in self.multi_asset.columns else None
    )
    if weight_series is not None and isinstance(weight_series.index, pd.MultiIndex):
        if "ticker" in weight_series.index.names:
            weight_series = weight_series.reset_index(level="ticker", drop=True)

    return plot_sample_concurrency(
        t1_series=t1_series,
        weight_series=weight_series,
        concurrency_threshold_pct=concurrency_threshold_pct,
    )


AssetOrganiser.plot_cusum_events = _ao_plot_cusum_events
AssetOrganiser.plot_sample_concurrency = _ao_plot_sample_concurrency


# --- IX02-05: StationaryTransformer ---
def _st_plot_stationarity_profile(self, raw_series, col, max_lags=40):
    from .features import plot_stationarity_profile
    from pyquantflow.data.features.fractional_differentiation import adf_screened_ffd

    d_star = self.optimal_d_.get(col, 1.0)
    ffd_series, _ = adf_screened_ffd(raw_series, d=d_star, thres=self.ffd_thres)
    return plot_stationarity_profile(
        raw_series, ffd_series, d_star, ticker=col, max_lags=max_lags
    )


StationaryTransformer.plot_stationarity_profile = _st_plot_stationarity_profile


# --- IX02-06: FeatureEvaluator ---
def _fe_plot_feature_clusters(self, df, regime_id=None):
    from .clustering import plot_feature_clusters

    all_features = self.features + self.raw_features
    corr_matrix = df[all_features].corr()

    # If importance_df exists, use it, otherwise call evaluate_importance (which returns the nested dict)
    if self.importance_df is not None:
        regime_results = self.importance_df
    else:
        regime_results = self.evaluate_importance(df)

    return plot_feature_clusters(
        regime_results=regime_results,
        correlation_matrix=corr_matrix,
        regime_id=regime_id,
    )


FeatureEvaluator.plot_feature_clusters = _fe_plot_feature_clusters


# --- IX02-07: Cross Validators ---
def _cv_plot_splits(self, X, y):
    from .cv import plot_cv_splits

    return plot_cv_splits(self, X, y)


PurgedKFoldCV.plot_splits = _cv_plot_splits
CombinatorialPurgedKFold.plot_splits = _cv_plot_splits


# --- IX02-08: PrimarySecondaryClassifier ---
def _psc_plot_meta_diagnostics(self, X, y_true):
    from .metalabel import plot_meta_label_entropy

    enriched = self.transform(X)
    if not hasattr(y_true, 'iloc'):
        y_true = pd.Series(y_true, index=enriched.index, name="label")
    else:
        y_true = y_true.rename("label")
        
    return plot_meta_label_entropy(
        enriched.join(y_true, how="left"),
    )


PrimarySecondaryClassifier.plot_meta_diagnostics = _psc_plot_meta_diagnostics


# --- IX02-09: GSADFTransformer ---
def _gsadf_plot_sadf_regimes(
    self, price_series, sadf_series, critical_value=1.4, events=None
):
    from .regimes import plot_sadf_regimes

    return plot_sadf_regimes(price_series, sadf_series, critical_value, events)


GSADFTransformer.plot_sadf_regimes = _gsadf_plot_sadf_regimes
