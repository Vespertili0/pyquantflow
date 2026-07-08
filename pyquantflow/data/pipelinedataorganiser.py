import pandas as pd
from typing import Dict, List, Optional

from .assetorganiser import AssetOrganiser
from .labels import BaseLabelFactory


class PipelineDataOrganiser:
    """
    Strict data provisioner for ML pipelines.

    Replicates panel alignment, label generation, and CUSUM down-sampling
    from ``AssetOrganiser``, but deliberately withholds any statistical
    transformation (no ``StationaryTransformer``).  Its sole output via
    :meth:`get_classifierengine_payload` is the raw, unscaled design
    matrix (``X_train``, ``X_test``), target arrays (``y``), sample
    weights (``w``), and embargo limits (``t1``).

    This class uses **composition** — it delegates data-wrangling work
    to an internal ``AssetOrganiser`` instance so the two classes stay
    loosely coupled while sharing tested panel-alignment logic.

    Parameters
    ----------
    data_map : Optional[Dict[str, pd.DataFrame]]
        Dictionary mapping tickers to their respective DataFrames.
    cutoff_date : str
        The date string (e.g., 'YYYY-MM-DD') separating train and test
        sets.
    target_features : List[str]
        List of column names to be used as targets (y).
    weight_col : Optional[str]
        Optional column name in the DataFrame containing target weights.
    multi_asset : Optional[pd.DataFrame]
        Pre-constructed multi-asset DataFrame.
    label_factory : Optional[BaseLabelFactory]
        Factory for generating labels and weights.
    """

    def __init__(
        self,
        data_map: Optional[Dict[str, pd.DataFrame]] = None,
        cutoff_date: Optional[str] = None,
        target_features: Optional[List[str]] = None,
        weight_col: Optional[str] = None,
        multi_asset: Optional[pd.DataFrame] = None,
        label_factory: Optional[BaseLabelFactory] = None,
    ) -> None:
        self._organiser = AssetOrganiser(
            data_map=data_map,
            cutoff_date=cutoff_date,
            target_features=target_features,
            weight_col=weight_col,
            multi_asset=multi_asset,
            label_factory=label_factory,
        )

    # ------------------------------------------------------------------
    # Read-only property pass-throughs
    # ------------------------------------------------------------------

    @property
    def multi_asset(self) -> Optional[pd.DataFrame]:
        """The full multi-asset panel DataFrame (read-only)."""
        return self._organiser.multi_asset

    @property
    def multi_asset_train(self) -> Optional[pd.DataFrame]:
        """Training partition of the multi-asset panel (read-only)."""
        return self._organiser.multi_asset_train

    @property
    def multi_asset_test(self) -> Optional[pd.DataFrame]:
        """Test partition of the multi-asset panel (read-only)."""
        return self._organiser.multi_asset_test

    @property
    def weight_col(self) -> Optional[str]:
        """The sample-weight column name (read-only)."""
        return self._organiser.weight_col

    @property
    def target_features(self) -> List[str]:
        """The target column names (read-only)."""
        return self._organiser.target_features

    # ------------------------------------------------------------------
    # Delegated data-wrangling methods
    # ------------------------------------------------------------------

    def prepare_multi_asset_frame(self) -> None:
        """Delegates to ``AssetOrganiser.prepare_multi_asset_frame``."""
        self._organiser.prepare_multi_asset_frame()

    def downsample_to_events(
        self,
        events: "pd.DatetimeIndex | list | set | Dict[str, pd.DatetimeIndex]",
    ) -> None:
        """Delegates to ``AssetOrganiser.downsample_to_events``."""
        self._organiser.downsample_to_events(events)

    def downsample_to_cusum_events(
        self,
        target_events_train: "int | Dict[str, int]",
        filter_col: str,
        vol_col: Optional[str] = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
    ) -> Dict[str, float]:
        """Delegates to ``AssetOrganiser.downsample_to_cusum_events``."""
        return self._organiser.downsample_to_cusum_events(
            target_events_train=target_events_train,
            filter_col=filter_col,
            vol_col=vol_col,
            span=span,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_step=alpha_step,
        )

    def apply_continuous_labels(self, price_col: str = "Close") -> None:
        """Delegates to ``AssetOrganiser.apply_continuous_labels``."""
        self._organiser.apply_continuous_labels(price_col=price_col)

    def apply_sample_weights(self, price_col: str = "Close") -> None:
        """Delegates to ``AssetOrganiser.apply_sample_weights``."""
        self._organiser.apply_sample_weights(price_col=price_col)

    def build_learning_pipeline(
        self,
        target_events_train: "int | Dict[str, int]",
        filter_col: str,
        price_col: str = "Close",
        vol_col: Optional[str] = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
    ) -> Dict[str, float]:
        """Delegates to ``AssetOrganiser.build_learning_pipeline``."""
        return self._organiser.build_learning_pipeline(
            target_events_train=target_events_train,
            filter_col=filter_col,
            price_col=price_col,
            vol_col=vol_col,
            span=span,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_step=alpha_step,
        )

    # ------------------------------------------------------------------
    # Own payload method — guarantees raw, untransformed output
    # ------------------------------------------------------------------

    def get_classifierengine_payload(
        self,
        features: List[str],
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, "pd.DataFrame | List[str] | str | None"]:
        """
        Extracts the prepared data and metadata into a dictionary suitable
        for unpacking (``**kwargs``) directly into
        ``ClassifierEngine.run_pipeline``.

        Semantically identical to ``AssetOrganiser.get_classifierengine_payload``
        but *guarantees* that the returned DataFrames are **raw and
        untransformed** — no ``StationaryTransformer`` or z-score scaling
        has been applied.

        Parameters
        ----------
        features : List[str]
            List of column names to be used as features.
        tickers : Optional[List[str]]
            Optional list of tickers to filter the returned datasets.
            If None, the full multi-asset DataFrame is returned.

        Returns
        -------
        Dict[str, pd.DataFrame | List[str] | str | None]
            The payload dictionary containing ``X_train``, ``y_train``,
            ``X_test``, ``y_test``, ``features``, and ``weight_col``.
        """
        return self._organiser.get_classifierengine_payload(
            features=features,
            tickers=tickers,
        )
