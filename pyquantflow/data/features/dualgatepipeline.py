from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from pyquantflow.data.assetorganiser import AssetOrganiser
from pyquantflow.model.feature_evaluation import FeatureEvaluator


class DualGatePipelineFactory:
    """
    Orchestrates the synchronisation and execution of the Two-Track Dual-Gate pipeline.
    Handles continuous feature evaluation alongside discrete price-action labelling.
    """

    def __init__(
        self,
        price_col: str = "Close",
        filter_col: str = "Close",
        vol_col: Optional[str] = None,
    ):
        """
        Initialises the DualGatePipelineFactory.

        Parameters
        ----------
        price_col : str, default "Close"
            Name of the column containing price data for labelling.
        filter_col : str, default "Close"
            Name of the column used for event filtering (e.g., CUSUM).
        vol_col : Optional[str], default None
            Optional pre-calculated volatility column name.
        """
        self.price_col = price_col
        self.filter_col = filter_col
        self.vol_col = vol_col

    def execute(
        self,
        organiser: AssetOrganiser,
        evaluator: FeatureEvaluator,
        target_events_train: Optional[int | Dict[str, int]] = None,
        target_labels: List[str] = ["label", "t1", "weight"],
        span: int = 50,
        estimator: Optional[BaseEstimator] = None,
        balance_classes: bool = True,
        greater_is_better: bool = False,
        needs_proba: bool = True,
        objective: str = "budget",
        t1_col: Optional[str] = None,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
    ) -> Tuple[AssetOrganiser, List[str]]:
        """
        Executes the Two-Track pipeline: continuous transformations, discrete event
        downsampling, alignment merge, and Gate 1 memory validation.

        Parameters
        ----------
        organiser : AssetOrganiser
            The organiser holding the multi-asset DataFrame to be processed.
        evaluator : FeatureEvaluator
            The feature evaluator that handles stationarity checks and pruning.
        target_labels : List[str], default ["label", "t1", "weight"]
            The column names for target metadata labels to isolate and synchronise.
        target_events_train : Optional[int | Dict[str, int]], default None
            Target event budget for CUSUM downsampling on the training set.
            Required when ``objective="budget"``; ignored when ``objective="uniqueness"``.
        span : int, default 50
            EWMA span for calculating dynamic volatility.
        estimator : Optional[BaseEstimator], default None
            A custom machine learning estimator to evaluate feature importance.
            If None, defaults to HistGradientBoostingClassifier().
        objective : str, default ``"budget"``
            CUSUM calibration objective forwarded to
            ``AssetOrganiser.build_learning_pipeline``.
            ``"budget"`` minimises distance to ``target_events_train``;
            ``"uniqueness"`` maximises average sample uniqueness of the
            generated events (requires ``t1_col``).
        t1_col : Optional[str], default None
            Column name containing ``t1`` barrier timestamps. Required when
            ``objective="uniqueness"``.
        alpha_min : float, default 0.5
            Minimum alpha multiplier for the CUSUM calibration grid search.
        alpha_max : float, default 3.0
            Maximum alpha multiplier for the CUSUM calibration grid search.
        alpha_step : float, default 0.1
            Step size for the CUSUM calibration grid search.

        Returns
        -------
        Tuple[AssetOrganiser, List[str]]
            A tuple containing:
            - The updated AssetOrganiser containing the downsampled and stationarised dataset.
            - A list of feature names that survived the Gate 1 memory check.
        """
        if organiser.multi_asset is None:
            organiser.prepare_multi_asset_frame()

        # Track original *transformed* features (these will be replaced in the organiser)
        original_features = list(evaluator.features)

        # TRACK B (Part 1): Snapshot the Unbroken Continuous Timeline
        # Include raw_features alongside transform-path features so they travel through
        # the synchronisation bridge. Raw features are never passed to replace_features,
        # so their original values in multi_asset are preserved by AssetOrganiser.
        all_eval_cols = evaluator.features + (evaluator.raw_features or [])
        continuous_df = organiser.multi_asset[all_eval_cols].copy()

        # TRACK A: Structural Label & Event Generation (Pure Price Action)
        organiser.build_learning_pipeline(
            target_events_train=target_events_train,
            filter_col=self.filter_col,
            price_col=self.price_col,
            vol_col=self.vol_col,
            span=span,
            objective=objective,
            t1_col=t1_col,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_step=alpha_step,
        )

        # Isolate the sparse event timestamps generated by Track A
        labeled_events_df = organiser.multi_asset[target_labels]
        cusum_index = labeled_events_df.index

        # THE SYNCHRONISATION BRIDGE: Map Sparse Metadata back to Unbroken Tracks
        continuous_df = continuous_df.join(labeled_events_df)

        # TRACK B (Part 2): Stationarity Transformations & Gate 1 Pruning
        # fit_transform_features dynamically drops features that fail the memory threshold
        continuous_transformed_df = evaluator.fit_transform_features(continuous_df)

        # Release continuous_df as it is no longer required
        del continuous_df

        # STATE RE-INJECTION: Downsample and Update the AssetOrganiser
        # Slice only the valid CUSUM event milestones out of the transformed matrix
        merged_df = continuous_transformed_df.loc[cusum_index]
        del continuous_transformed_df

        # Drop any remaining NaN rows generated by rolling operations or padding
        merged_df = merged_df.dropna()

        if estimator is None:
            estimator = HistGradientBoostingClassifier()

        evaluator.evaluate_importance(
            df=merged_df,
            estimator=estimator,
            metric=log_loss,
            metric_kwargs={
                "labels": np.unique(merged_df[evaluator.target_col].astype(int).values)
            },
            needs_proba=needs_proba,
            greater_is_better=greater_is_better,
            balance_classes=balance_classes,
        )

        # Commit the clean, stationary panel back into the organiser's state machine,
        # replacing only the features and preserving other original columns/rows.
        organiser.replace_features(merged_df, original_features)

        # Return the updated organiser and the curated list of survivor features
        return organiser, evaluator.importance_df
