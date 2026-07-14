import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from .utils import (
    align_and_ffill_multiasset,
    restructure_map_2_multiasset_df,
    pipe_indicator,
)
from .labels import get_cusum_events, calibrate_cusum_alpha, BaseLabelFactory
from .features.indicator import ICHIMOKU


class AssetOrganiser:
    """
    Organises and prepares multi-asset data for a quantitative classifier.

    This class handles the conversion of a dictionary of disparate asset DataFrames
    into an aligned multi-index DataFrame, splits it based on a cutoff date,
    and manages the fitting and transformation process using a specified classifier.
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
        """
        Initialises the AssetOrganiser.

        Args:
            data_map (Optional[Dict[str, pd.DataFrame]]): Dictionary mapping tickers to
                their respective DataFrames.
            cutoff_date (str): The date string (e.g., 'YYYY-MM-DD') separating
                train and test sets.
            target_features (List[str]): List of column names to be used as targets (y).
            weight_col (Optional[str]): Optional column name in the DataFrame
                containing target weights.
            multi_asset (Optional[pd.DataFrame]): Pre-constructed multi-asset DataFrame.
            label_factory (Optional[BaseLabelFactory]): Factory for generating labels and weights.
        """
        if data_map is None and multi_asset is None:
            raise ValueError("Either 'data_map' or 'multi_asset' must be provided.")
        if data_map is not None and multi_asset is not None:
            raise ValueError("Cannot provide both 'data_map' and 'multi_asset'.")
        if cutoff_date is None:
            raise ValueError("'cutoff_date' is required.")
        if target_features is None:
            raise ValueError("'target_features' is required.")

        self.data_map: Optional[Dict[str, pd.DataFrame]] = data_map
        self.cutoff_date: str = cutoff_date
        self.target_features: List[str] = target_features
        self.weight_col: Optional[str] = weight_col
        self.label_factory: Optional[BaseLabelFactory] = label_factory

        self.multi_asset: Optional[pd.DataFrame] = multi_asset
        self.multi_asset_train: Optional[pd.DataFrame] = None
        self.multi_asset_test: Optional[pd.DataFrame] = None

        if self.multi_asset is not None:
            self._split_train_test()

    def _split_train_test(self) -> None:
        """
        Splits the multi_asset DataFrame into train and test sets
        based on the cutoff date.
        """
        datetime_vals = pd.to_datetime(
            self.multi_asset.index.get_level_values("datetime"), utc=True
        )
        cutoff = pd.to_datetime(self.cutoff_date, utc=True)

        self.multi_asset_train = self.multi_asset[datetime_vals < cutoff]
        self.multi_asset_test = self.multi_asset[datetime_vals >= cutoff]

    def prepare_multi_asset_frame(self) -> None:
        """
        Converts data_map to Date-Ticker multi-index DataFrame or splits multi_asset if already provided.
        """
        if self.data_map is not None:
            self.multi_asset = align_and_ffill_multiasset(
                restructure_map_2_multiasset_df(self.data_map)
            )
        self._split_train_test()

        return None

    def downsample_to_events(
        self,
        events: pd.DatetimeIndex | list | set | Dict[str, pd.DatetimeIndex],
    ) -> None:
        """
        Down-samples the multi-asset DataFrame to keep only the dates matching
        the specified events for each ticker.

        Parameters
        ----------
        events : pd.DatetimeIndex | list | set | Dict[str, pd.DatetimeIndex]
            A DatetimeIndex, list, or set of timestamps to down-sample to globally,
            or a dictionary mapping ticker symbols to their specific event DatetimeIndexes.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()

        datetimes = self.multi_asset.index.get_level_values("datetime")

        if isinstance(events, (pd.DatetimeIndex, list, set)):
            event_set = {pd.Timestamp(dt) for dt in events}
            mask = [pd.Timestamp(dt) in event_set for dt in datetimes]
        elif isinstance(events, dict):
            event_sets = {
                tk: {pd.Timestamp(dt) for dt in idx} for tk, idx in events.items()
            }
            tickers = self.multi_asset.index.get_level_values("ticker")
            mask = [
                pd.Timestamp(dt) in event_sets[tk] if tk in event_sets else False
                for dt, tk in zip(datetimes, tickers)
            ]
        else:
            raise TypeError(
                "events must be a DatetimeIndex, list, set, or dict of ticker to DatetimeIndex."
            )

        self.multi_asset = self.multi_asset[mask]
        self._split_train_test()

        return None

    def downsample_to_cusum_events(
        self,
        target_events_train: int | Dict[str, int],
        filter_col: str,
        vol_col: Optional[str] = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
        objective: str = "budget",
        t1_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calibrates optimal alpha scalars on the training set and down-samples the
        multi-asset DataFrame using causal dynamic thresholds.

        Parameters
        ----------
        target_events_train : int | Dict[str, int]
            The target event count for the training fold. If int, applied to all tickers.
            If dict, maps ticker to specific target count.
            Required when ``objective="budget"``.
        filter_col : str
            The name of the column in the DataFrame to run CUSUM on.
        vol_col : Optional[str], default=None
            Optional pre-calculated volatility column name. If None, dynamic EWMA
            volatility is calculated on filter_col.
        span : int, default=100
            The EWMA span for calculating dynamic volatility.
        alpha_min : float, default=0.5
            Minimum alpha multiplier.
        alpha_max : float, default=3.0
            Maximum alpha multiplier.
        alpha_step : float, default=0.1
            Grid search step size.
        objective : str, default="budget"
            Calibration objective. ``"budget"`` minimises the distance to
            ``target_events_train``; ``"uniqueness"`` maximises average sample
            uniqueness of the generated events (requires ``t1_col``).
        t1_col : Optional[str], default=None
            Column name containing ``t1`` barrier timestamps. Required when
            ``objective="uniqueness"``.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping ticker symbols to their calibrated optimal alpha values.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()

        tickers = self.multi_asset.index.get_level_values("ticker").unique()
        calibrated_alphas = {}

        # 1. Calibrate on the training set strictly (Pipeline Isolation)
        if self.multi_asset_train is None:
            raise ValueError(
                "Training data is not prepared. Call prepare_multi_asset_frame() first."
            )

        for tk in tickers:
            # Check target events budget
            if isinstance(target_events_train, dict):
                if tk not in target_events_train:
                    raise KeyError(f"Ticker '{tk}' not found in target_events_train.")
                target = target_events_train[tk]
            else:
                target = int(target_events_train)

            # Extract training series/volatility for ticker
            if tk not in self.multi_asset_train.index.get_level_values("ticker"):
                # If ticker has no data in training fold, use default alpha
                calibrated_alphas[tk] = alpha_min
                continue

            ticker_train_df = self.multi_asset_train.xs(tk, level="ticker")
            ticker_train_series = ticker_train_df[filter_col]

            ticker_train_vol = None
            if vol_col:
                try:
                    ticker_train_vol = ticker_train_df[vol_col]
                except KeyError:
                    pass

            # Extract t1 series if uniqueness objective is requested
            ticker_train_t1 = None
            if objective == "uniqueness" and t1_col is not None:
                if t1_col not in ticker_train_df.columns:
                    raise KeyError(
                        f"Column '{t1_col}' (t1_col) not found in the training data "
                        f"for ticker '{tk}'. Available columns: "
                        f"{list(ticker_train_df.columns)}. "
                        "Ensure apply_continuous_labels() has been called before "
                        "downsample_to_cusum_events() when using objective='uniqueness'."
                    )
                ticker_train_t1 = ticker_train_df[t1_col]

            # Run calibration strictly on training set series
            alpha = calibrate_cusum_alpha(
                series=ticker_train_series,
                target_events=target,
                volatility=ticker_train_vol,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                alpha_step=alpha_step,
                span=span,
                objective=objective,
                t1=ticker_train_t1,
            )
            calibrated_alphas[tk] = alpha

        # 2. Run CUSUM filter with dynamic, volatility-adjusted threshold
        # using the calibrated/frozen alphas across the ENTIRE dataset (no leakage)
        events_map = {}
        for tk in tickers:
            alpha = calibrated_alphas[tk]
            series_all = self.multi_asset.xs(tk, level="ticker")[filter_col]

            # Use pre-calculated volatility if specified
            if vol_col:
                try:
                    vol_all = self.multi_asset.xs(tk, level="ticker")[vol_col]
                except KeyError:
                    vol_all = series_all.ewm(span=span).std()
            else:
                vol_all = series_all.ewm(span=span).std()

            threshold_all = alpha * vol_all

            # Filter events
            events = get_cusum_events(series_all, threshold_all)
            events_map[tk] = events

        # 3. Down-sample the organiser's multi-asset DataFrame using these events
        self.downsample_to_events(events_map)

        return calibrated_alphas

    def apply_continuous_labels(self, price_col: str = "Close") -> None:
        """
        Applies the label_factory strictly on the continuous, un-sampled price series.
        Injects the resulting 'label' and 't1' columns into the multi_asset DataFrame.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()

        if self.label_factory is None:
            raise ValueError("No label_factory provided.")

        tickers = self.multi_asset.index.get_level_values("ticker").unique()
        all_labels = []

        for tk in tickers:
            ticker_df = self.multi_asset.xs(tk, level="ticker")
            labels_df = self.label_factory.generate_labels(
                ticker_df, price_col=price_col
            )

            # Ensure the index has a name so that it resets to "datetime" properly
            if labels_df.index.name is None:
                labels_df.index.name = "datetime"

            # Add ticker level back to index for concatenation
            labels_df["ticker"] = tk
            labels_df = labels_df.reset_index().set_index(["datetime", "ticker"])
            all_labels.append(labels_df)

        labels_concat = pd.concat(all_labels)

        # Drop existing label/t1 columns if they exist to prevent duplicates
        drop_cols = [c for c in labels_concat.columns if c in self.multi_asset.columns]
        if drop_cols:
            self.multi_asset = self.multi_asset.drop(columns=drop_cols)

        # Merge labels back into multi_asset
        self.multi_asset = self.multi_asset.join(labels_concat, how="left")

        # Drop rows with NaN/NaT values in the label-related columns
        self.multi_asset = self.multi_asset.dropna(subset=labels_concat.columns)
        self._split_train_test()

    def apply_sample_weights(self, price_col: str = "Close") -> None:
        """
        Calculates sample weights strictly on the currently filtered multi_asset DataFrame.
        This must be run AFTER down-sampling (e.g., CUSUM) to correctly calculate concurrency.
        """
        if self.multi_asset is None:
            raise ValueError("Multi-asset DataFrame not initialized.")

        if self.label_factory is None:
            raise ValueError("No label_factory provided.")

        if "t1" not in self.multi_asset.columns:
            raise KeyError(
                "The column 't1' is missing. Please run apply_continuous_labels() first."
            )

        tickers = self.multi_asset.index.get_level_values("ticker").unique()
        all_weights = []

        for tk in tickers:
            ticker_df = self.multi_asset.xs(tk, level="ticker")
            t1 = ticker_df["t1"]
            returns = ticker_df[price_col].pct_change()

            weights = self.label_factory.generate_weights(t1, returns)
            weights.name = self.weight_col if self.weight_col else "weight"

            weights_df = weights.to_frame()
            weights_df["ticker"] = tk
            weights_df = weights_df.reset_index().set_index(["datetime", "ticker"])
            all_weights.append(weights_df)

        weights_concat = pd.concat(all_weights)

        # Globally rescale weights so their mean is 1.0
        if weights_concat[self.weight_col if self.weight_col else "weight"].sum() > 0:
            weights_concat[self.weight_col if self.weight_col else "weight"] = (
                weights_concat[self.weight_col if self.weight_col else "weight"]
                / weights_concat[
                    self.weight_col if self.weight_col else "weight"
                ].mean()
            )

        # Clip the extreme tails to prevent overfitting and zero-weights
        # Lower bound of 0.01 keeps highly concurrent events barely visible.
        # Upper bound caps the maximum influence of a single event to a
        # safe multiple (e.g., 10x or the 99th percentile).
        upper_cap = weights_concat[
            self.weight_col if self.weight_col else "weight"
        ].quantile(0.99)
        weights_concat[self.weight_col if self.weight_col else "weight"] = (
            weights_concat[self.weight_col if self.weight_col else "weight"].clip(
                lower=0.01, upper=upper_cap
            )
        )

        col_name = self.weight_col if self.weight_col else "weight"
        if col_name in self.multi_asset.columns:
            self.multi_asset = self.multi_asset.drop(columns=[col_name])

        self.multi_asset = self.multi_asset.join(weights_concat, how="left")

        # Drop rows with NaN values in the weights column
        self.multi_asset = self.multi_asset.dropna(subset=[col_name])

        # If no explicit weight_col was passed during __init__, update it so the pipeline knows
        if not self.weight_col:
            self.weight_col = col_name

        self._split_train_test()

    def build_learning_pipeline(
        self,
        target_events_train: int | Dict[str, int],
        filter_col: str,
        price_col: str = "Close",
        vol_col: Optional[str] = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
        objective: str = "budget",
        t1_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Orchestrates the preparation pipeline to strictly prevent sequential data hazards:
        1. Computes continuous labels (avoiding look-ahead scaling errors).
        2. Down-samples the dataset based on dynamic CUSUM events.
        3. Calculates sample weights based on the surviving active bets (concurrency).

        Parameters
        ----------
        objective : str, default="budget"
            CUSUM calibration objective forwarded to ``downsample_to_cusum_events``.
            ``"budget"`` minimises distance to ``target_events_train``;
            ``"uniqueness"`` maximises average sample uniqueness.
        t1_col : Optional[str], default=None
            Column name for ``t1`` barrier timestamps. Required when
            ``objective="uniqueness"``.

        Returns
        -------
        Dict[str, float]
            Calibrated alpha thresholds from the CUSUM down-sampling phase.
        """
        self.apply_continuous_labels(price_col=price_col)

        alphas = self.downsample_to_cusum_events(
            target_events_train=target_events_train,
            filter_col=filter_col,
            vol_col=vol_col,
            span=span,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_step=alpha_step,
            objective=objective,
            t1_col=t1_col,
        )

        self.apply_sample_weights(price_col=price_col)

        return alphas

    def add_model_predictions(
        self,
        model: BaseEstimator,
        features: List[str],
        prefix: str = "primary",
        filter_prediction: Optional[int] = None,
    ) -> None:
        """
        Fits the model on the multiasset data.
        Generates predictions and probability entropy from the provided model,
        injects them into the multi_asset DataFrame, and optionally filters the dataset.

        Args:
            model: A fitted Scikit-Learn estimator.
            features: List of column names to pass to the model.
            filter_prediction: Optional prediction value (e.g., 1) to filter the
                                resulting dataset (Meta-Labelling).
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()

        X = self.multi_asset[features]
        preds = model.predict(X)
        probas = model.predict_proba(X)

        # Calculate Shannon Entropy: H = -sum(p * log(p))
        prob_entropy = entropy(probas, axis=1)

        # Inject predictions as new features out-of-place
        new_columns = pd.DataFrame(
            {f"{prefix}_pred": preds, f"{prefix}_entropy": prob_entropy},
            index=self.multi_asset.index,
        )

        # Create a DataFrame for the probabilities with dynamic column names
        # This automatically handles N classes (primary_proba0, primary_proba1, etc.)
        proba_df = pd.DataFrame(
            probas,
            columns=[f"{prefix}_proba{i}" for i in range(probas.shape[1])],
            index=self.multi_asset.index,
        )

        # Concatenate them together
        new_columns = pd.concat([new_columns, proba_df], axis=1)

        self.multi_asset = pd.concat([self.multi_asset, new_columns], axis=1)
        if filter_prediction is not None:
            self.multi_asset = self.multi_asset[
                self.multi_asset[f"{prefix}_pred"] == filter_prediction
            ]

        self._split_train_test()

    def get_classifierengine_payload(
        self,
        features: List[str],
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame | List[str] | str | None]:
        """
        Extracts the prepared data and metadata into a dictionary suitable for
        unpacking (**kwargs) directly into `ClassifierEngine.run_pipeline`.

        Args:
            features (List[str]): List of column names to be used as features.
            tickers (Optional[List[str]]): Optional list of tickers to filter the returned datasets.
                If None, the full multi-asset DataFrame is returned.

        Returns:
            Dict[str, pd.DataFrame | List[str] | str | None]: The payload dictionary.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            self.prepare_multi_asset_frame()

        # Remove weight_col from features list to strictly separate metadata
        features_copy = list(features)
        if self.weight_col and self.weight_col in features_copy:
            features_copy.remove(self.weight_col)

        X_train = self.multi_asset_train
        X_test = self.multi_asset_test

        if tickers is not None:
            X_train = X_train[X_train.index.get_level_values("ticker").isin(tickers)]
            X_test = X_test[X_test.index.get_level_values("ticker").isin(tickers)]

        return {
            "X_train": X_train,
            "y_train": X_train[self.target_features],
            "X_test": X_test,
            "y_test": X_test[self.target_features],
            "features": features_copy,
            "weight_col": self.weight_col,
        }

    def to_tsfeatures_format(
        self,
        value_col: str,
        subset: str = "all",
    ) -> pd.DataFrame:
        """
        Transforms the multi-asset DataFrame into the format required by Nixtla's `tsfeatures`.

        The output DataFrame will contain the columns:
        - `unique_id`: mapped from the 'ticker' index level.
        - `ds`: mapped from the 'datetime' index level.
        - `y`: mapped from the specified value_col.

        Args:
            value_col (str): The column name to extract as target 'y' (e.g., 'Close').
            subset (str): The data subset to transform: 'all', 'train', or 'test'.

        Returns:
            pd.DataFrame: Long-format DataFrame ready for tsfeatures analysis.
        """
        if subset == "all":
            df = self.multi_asset
            if df is None:
                self.prepare_multi_asset_frame()
                df = self.multi_asset
        elif subset == "train":
            df = self.multi_asset_train
            if df is None:
                self.prepare_multi_asset_frame()
                df = self.multi_asset_train
        elif subset == "test":
            df = self.multi_asset_test
            if df is None:
                self.prepare_multi_asset_frame()
                df = self.multi_asset_test
        else:
            raise ValueError(
                f"Unknown subset: {subset}. Must be 'all', 'train', or 'test'."
            )

        if df is None:
            raise ValueError("No multi-asset DataFrame available to transform.")

        if value_col not in df.columns:
            raise KeyError(f"Column '{value_col}' not found in the DataFrame.")

        # Reset index to extract datetime and ticker levels
        df_reset = df.reset_index()

        # Rename to match tsfeatures requirements:
        # unique_id: time series identifier (ticker)
        # ds: datetimes
        # y: value column
        df_ts = df_reset.rename(
            columns={
                "ticker": "unique_id",
                "datetime": "ds",
                value_col: "y",
            }
        )

        return df_ts[["unique_id", "ds", "y"]].copy()

    def apply_ichimoku_regime(self) -> None:
        """
        Computes the Ichimoku Cloud and injects a binary ``ichimoku_regime``
        column into ``self.multi_asset``, grouped by ticker.

        The regime is ``1`` when the closing price is above *both* the shifted
        Senkou Span A and Senkou Span B (i.e., the price is above the cloud),
        and ``0`` otherwise.  Warm-up bars where the cloud spans are NaN
        (roughly the first 78 periods) are also set to ``0``.

        Raw Ichimoku component columns are dropped immediately after the mask
        is created to keep the feature matrix clean.  **No rows are dropped.**
        The method mutates ``self.multi_asset`` in-place and calls
        ``_split_train_test()`` at the end.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()

        _ICHIMOKU_COLS = [
            "tenkan_sen",
            "kijun_sen",
            "span_a",
            "span_b",
            "span_a_shifted",
            "span_b_shifted",
        ]
        _ICHIMOKU_OUTPUT_NAMES = _ICHIMOKU_COLS + [None]  # chikou skipped

        tickers = self.multi_asset.index.get_level_values("ticker").unique()
        all_regime = []

        for tk in tickers:
            ticker_df = self.multi_asset.xs(tk, level="ticker").copy()

            # Compute all Ichimoku components via pipe_indicator
            ticker_df = pipe_indicator(
                ticker_df,
                ICHIMOKU,
                input_map={"high": "High", "low": "Low"},
                output_names=_ICHIMOKU_OUTPUT_NAMES,
            )

            # Build binary regime: 1 if Close is above both shifted spans
            above_cloud = (ticker_df["Close"] > ticker_df["span_a_shifted"]) & (
                ticker_df["Close"] > ticker_df["span_b_shifted"]
            )
            # Fill warm-up NaNs (first ~78 bars) with 0 (no regime)
            ticker_df["ichimoku_regime"] = above_cloud.fillna(False).astype(int)

            # Drop raw Ichimoku components to keep the feature matrix clean
            ticker_df = ticker_df.drop(columns=_ICHIMOKU_COLS, errors="ignore")

            # Rebuild multi-index for concatenation
            regime_df = ticker_df[["ichimoku_regime"]].copy()
            regime_df.index.name = "datetime"
            regime_df["ticker"] = tk
            regime_df = regime_df.reset_index().set_index(["datetime", "ticker"])
            all_regime.append(regime_df)

        regime_concat = pd.concat(all_regime)

        # Drop existing column if present to prevent duplicates
        if "ichimoku_regime" in self.multi_asset.columns:
            self.multi_asset = self.multi_asset.drop(columns=["ichimoku_regime"])

        self.multi_asset = self.multi_asset.join(regime_concat, how="left")
        # Any remaining NaNs (e.g. misaligned index edge cases) default to 0
        self.multi_asset["ichimoku_regime"] = (
            self.multi_asset["ichimoku_regime"].fillna(0).astype(int)
        )
        self._split_train_test()

    def update_multi_asset(self, df: pd.DataFrame) -> None:
        """
        Overwrites the internal multi_asset panel dataset with engineered features
        and automatically re-synchronises the train and test split boundaries.
        """
        if df.index.names != ["datetime", "ticker"]:
            raise ValueError(
                "DataFrame index must match MultiIndex format ['datetime', 'ticker']."
            )

        self.multi_asset = df.copy()
        self._split_train_test()

    def replace_features(
        self, transformed_df: pd.DataFrame, original_features: List[str]
    ) -> None:
        """
        Replaces the original features in the multi_asset panel dataset with
        transformed features, drops features that failed the pruning step,
        aligns the dataset to the transformed dataset's index (removing rows
        dropped during transformation), and re-synchronises the train/test split boundaries.
        """
        if transformed_df.index.names != ["datetime", "ticker"]:
            raise ValueError(
                "DataFrame index must match MultiIndex format ['datetime', 'ticker']."
            )
        if self.multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        # Align to transformed_df index (downsampled and dropna'd rows)
        self.multi_asset = self.multi_asset.loc[transformed_df.index].copy()

        # Identify features that were kept and those that were dropped
        surviving_features = [
            f for f in original_features if f in transformed_df.columns
        ]
        failed_features = [
            f for f in original_features if f not in transformed_df.columns
        ]

        # Replace surviving features with transformed versions
        for feat in surviving_features:
            self.multi_asset[feat] = transformed_df[feat]

        # Drop failed features
        self.multi_asset = self.multi_asset.drop(
            columns=failed_features, errors="ignore"
        )

        # Re-synchronise train and test splits
        self._split_train_test()
