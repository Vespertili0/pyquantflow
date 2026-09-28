"""
Asset Organiser Module

This module provides the AssetOrganiser class for preparing, aligning, and transforming
multi-asset panel data suitable for quantitative machine learning pipelines. It orchestrates
continuous labelling, dynamic CUSUM down-sampling, feature generation (such as Ichimoku regimes),
and sample weight calculation while strictly preventing sequential data hazards.
"""

import pandas as pd
from scipy.stats import entropy
from sklearn.base import BaseEstimator

from .features.indicator import ICHIMOKU
from .labels import BaseLabelFactory, calibrate_cusum_alpha, get_cusum_events
from .utils import (
    align_and_ffill_multiasset,
    pipe_indicator,
    restructure_map_2_multiasset_df,
)


class PanelBuilder:
    def __init__(self, cutoff_date: str) -> None:
        self.cutoff_date = cutoff_date

    def split_train_test(
        self, multi_asset: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Splits the multi_asset DataFrame into train and test sets
        based on the cutoff date.
        """
        if multi_asset is None:
            return None, None
        if (
            not isinstance(multi_asset.index, pd.MultiIndex)
            or "datetime" not in multi_asset.index.names
        ):
            return multi_asset, multi_asset
        datetime_vals = pd.to_datetime(
            multi_asset.index.get_level_values("datetime"), utc=True
        )
        cutoff = pd.to_datetime(self.cutoff_date, utc=True)

        multi_asset_train = multi_asset[datetime_vals < cutoff]
        multi_asset_test = multi_asset[datetime_vals >= cutoff]
        return multi_asset_train, multi_asset_test

    def prepare(
        self, data_map: dict | None, multi_asset: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """
        Converts data_map to Date-Ticker multi-index DataFrame or splits multi_asset if already provided.
        """
        if data_map is not None:
            return align_and_ffill_multiasset(restructure_map_2_multiasset_df(data_map))
        return multi_asset

    def update(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Overwrites the internal multi_asset panel dataset with engineered features
        and automatically re-synchronises the train and test split boundaries.
        """
        if df.index.names != ["datetime", "ticker"]:
            raise ValueError(
                "DataFrame index must match MultiIndex format ['datetime', 'ticker']."
            )
        return df.copy()

    def replace_features(
        self,
        multi_asset: pd.DataFrame,
        transformed_df: pd.DataFrame,
        original_features: list[str],
    ) -> pd.DataFrame:
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
        if multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        # Align to transformed_df index (downsampled and dropna'd rows)
        multi_asset_new = multi_asset.loc[transformed_df.index].copy()

        # Identify features that were kept and those that were dropped
        surviving_features = [
            f for f in original_features if f in transformed_df.columns
        ]
        failed_features = [
            f for f in original_features if f not in transformed_df.columns
        ]

        # Replace surviving features with transformed versions
        for feat in surviving_features:
            multi_asset_new[feat] = transformed_df[feat]

        # Drop failed features
        multi_asset_new = multi_asset_new.drop(columns=failed_features, errors="ignore")
        return multi_asset_new

    def to_tsfeatures_format(
        self,
        multi_asset: pd.DataFrame | None,
        multi_asset_train: pd.DataFrame | None,
        multi_asset_test: pd.DataFrame | None,
        value_col: str,
        subset: str = "all",
    ) -> pd.DataFrame:
        """
        Transforms the multi-asset DataFrame into the format required by Nixtla's `tsfeatures`.
        """
        if subset == "all":
            df = multi_asset
        elif subset == "train":
            df = multi_asset_train
        elif subset == "test":
            df = multi_asset_test
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

        df_ts = df_reset.rename(
            columns={
                "ticker": "unique_id",
                "datetime": "ds",
                value_col: "y",
            }
        )

        return df_ts[["unique_id", "ds", "y"]].copy()


class EventFilter:
    def __init__(self) -> None:
        pass

    def filter_to_events(
        self,
        multi_asset: pd.DataFrame,
        events: pd.DatetimeIndex | list | set | dict[str, pd.DatetimeIndex],
    ) -> pd.DataFrame:
        """
        Down-samples the multi-asset DataFrame to keep only the dates matching
        the specified events for each ticker.
        """
        if multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        datetimes = multi_asset.index.get_level_values("datetime")

        if isinstance(events, (pd.DatetimeIndex, list, set)):
            event_set = {pd.Timestamp(dt) for dt in events}
            mask = [pd.Timestamp(dt) in event_set for dt in datetimes]
        elif isinstance(events, dict):
            event_sets = {
                tk: {pd.Timestamp(dt) for dt in idx} for tk, idx in events.items()
            }
            tickers = multi_asset.index.get_level_values("ticker")
            mask = [
                pd.Timestamp(dt) in event_sets[tk] if tk in event_sets else False
                for dt, tk in zip(datetimes, tickers)
            ]
        else:
            raise TypeError(
                "events must be a DatetimeIndex, list, set, or dict of ticker to DatetimeIndex."
            )

        return multi_asset[mask].copy()

    def cusum_filter(
        self,
        multi_asset: pd.DataFrame,
        multi_asset_train: pd.DataFrame,
        target_events_train: int | dict[str, int],
        filter_col: str,
        vol_col: str | None = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
        objective: str = "budget",
        t1_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, pd.DatetimeIndex], dict[str, float]]:
        """
        Calibrates optimal alpha scalars on the training set and down-samples the
        multi-asset DataFrame using causal dynamic thresholds.
        """
        if multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        if objective == "uniqueness" and t1_col is None:
            raise ValueError(
                "objective='uniqueness' requires a valid t1_col, but t1_col=None was "
                "provided. Ensure apply_continuous_labels() has been called so that a "
                "t1 barrier column exists in the DataFrame, then pass its name via "
                "the t1_col argument."
            )

        tickers = multi_asset.index.get_level_values("ticker").unique()
        calibrated_alphas = {}

        if multi_asset_train is None:
            raise ValueError(
                "Training data is not prepared. Call prepare_multi_asset_frame() first."
            )

        for tk in tickers:
            if isinstance(target_events_train, dict):
                if tk not in target_events_train:
                    raise KeyError(f"Ticker '{tk}' not found in target_events_train.")
                target = target_events_train[tk]
            else:
                target = int(target_events_train)

            if tk not in multi_asset_train.index.get_level_values("ticker"):
                calibrated_alphas[tk] = alpha_min
                continue

            ticker_train_df = multi_asset_train.xs(tk, level="ticker")
            ticker_train_series = ticker_train_df[filter_col]

            ticker_train_vol = None
            if vol_col:
                try:
                    ticker_train_vol = ticker_train_df[vol_col]
                except KeyError:
                    pass

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

        events_map = {}
        for tk in tickers:
            alpha = calibrated_alphas[tk]
            series_all = multi_asset.xs(tk, level="ticker")[filter_col]

            if vol_col:
                try:
                    vol_all = multi_asset.xs(tk, level="ticker")[vol_col]
                except KeyError:
                    vol_all = series_all.ewm(span=span).std()
            else:
                vol_all = series_all.ewm(span=span).std()

            threshold_all = alpha * vol_all

            events = get_cusum_events(series_all, threshold_all)
            events_map[tk] = events

        filtered_multi_asset = self.filter_to_events(multi_asset, events_map)

        return filtered_multi_asset, events_map, calibrated_alphas


class LabelPipeline:
    def __init__(
        self,
        label_factory: BaseLabelFactory | None,
        weight_col: str,
        target_features: list[str],
    ) -> None:
        self.label_factory = label_factory
        self.weight_col = weight_col
        self.target_features = target_features

    def apply_continuous_labels(
        self, multi_asset: pd.DataFrame, price_col: str = "Close"
    ) -> pd.DataFrame:
        if multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        if self.label_factory is None:
            raise ValueError("No label_factory provided.")

        tickers = multi_asset.index.get_level_values("ticker").unique()
        all_labels = []

        for tk in tickers:
            ticker_df = multi_asset.xs(tk, level="ticker")
            labels_df = self.label_factory.generate_labels(
                ticker_df, price_col=price_col
            )

            if labels_df.index.name is None:
                labels_df.index.name = "datetime"

            labels_df["ticker"] = tk
            labels_df = labels_df.reset_index().set_index(["datetime", "ticker"])
            all_labels.append(labels_df)

        if not all_labels:
            return multi_asset

        labels_concat = pd.concat(all_labels)

        drop_cols = [c for c in labels_concat.columns if c in multi_asset.columns]
        if drop_cols:
            multi_asset = multi_asset.drop(columns=drop_cols)

        multi_asset = multi_asset.join(labels_concat, how="left")
        multi_asset = multi_asset.dropna(subset=labels_concat.columns)
        return multi_asset

    def apply_sample_weights(
        self, multi_asset: pd.DataFrame, price_col: str = "Close"
    ) -> pd.DataFrame:
        if multi_asset is None:
            raise ValueError("Multi-asset DataFrame not initialized.")

        if self.label_factory is None:
            raise ValueError("No label_factory provided.")

        if "t1" not in multi_asset.columns:
            raise KeyError(
                "The column 't1' is missing. Please run apply_continuous_labels() first."
            )

        tickers = multi_asset.index.get_level_values("ticker").unique()
        all_weights = []

        for tk in tickers:
            ticker_df = multi_asset.xs(tk, level="ticker")
            t1 = ticker_df["t1"]
            returns = ticker_df[price_col].pct_change()

            weights = self.label_factory.generate_weights(t1, returns)
            weights.name = self.weight_col

            weights_df = weights.to_frame()
            weights_df["ticker"] = tk
            weights_df = weights_df.reset_index().set_index(["datetime", "ticker"])
            all_weights.append(weights_df)

        if not all_weights:
            return multi_asset

        weights_concat = pd.concat(all_weights)

        if weights_concat[self.weight_col].sum() > 0:
            weights_concat[self.weight_col] = (
                weights_concat[self.weight_col] / weights_concat[self.weight_col].mean()
            )

        upper_cap = weights_concat[self.weight_col].quantile(0.99)
        weights_concat[self.weight_col] = weights_concat[self.weight_col].clip(
            lower=0.01, upper=upper_cap
        )

        if self.weight_col in multi_asset.columns:
            multi_asset = multi_asset.drop(columns=[self.weight_col])

        multi_asset = multi_asset.join(weights_concat, how="left")
        multi_asset = multi_asset.dropna(subset=[self.weight_col])
        return multi_asset

    def apply_ichimoku_regime(
        self, multi_asset: pd.DataFrame, mode: str = "standard", displacement: int = 26
    ) -> pd.DataFrame:
        if mode not in ("standard", "confirmed", "strict"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose from 'standard', 'confirmed', or 'strict'."
            )

        if multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        _ICHIMOKU_COLS = [
            "tenkan_sen",
            "kijun_sen",
            "span_a",
            "span_b",
            "span_a_shifted",
            "span_b_shifted",
        ]
        _ICHIMOKU_OUTPUT_NAMES = _ICHIMOKU_COLS + [None]

        tickers = multi_asset.index.get_level_values("ticker").unique()
        all_regime = []

        for tk in tickers:
            ticker_df = multi_asset.xs(tk, level="ticker").copy()

            ticker_df = pipe_indicator(
                ticker_df,
                ICHIMOKU,
                input_map={"high": "High", "low": "Low"},
                output_names=_ICHIMOKU_OUTPUT_NAMES,
            )

            above_cloud = (ticker_df["Close"] > ticker_df["span_a_shifted"]) & (
                ticker_df["Close"] > ticker_df["span_b_shifted"]
            )

            if mode == "standard":
                regime_mask = above_cloud
            elif mode == "confirmed":
                cloud_positive = ticker_df["span_a"] > ticker_df["span_b"]
                tk_momentum = ticker_df["tenkan_sen"] > ticker_df["kijun_sen"]
                regime_mask = above_cloud & cloud_positive & tk_momentum
            else:
                cloud_positive = ticker_df["span_a"] > ticker_df["span_b"]
                chikou_breakout = (
                    ticker_df["Close"] > ticker_df["span_a_shifted"].shift(displacement)
                ) & (
                    ticker_df["Close"] > ticker_df["span_b_shifted"].shift(displacement)
                )
                regime_mask = above_cloud & cloud_positive & chikou_breakout

            ticker_df["ichimoku_regime"] = regime_mask.fillna(False).astype(int)
            ticker_df = ticker_df.drop(columns=_ICHIMOKU_COLS, errors="ignore")

            regime_df = ticker_df[["ichimoku_regime"]].copy()
            regime_df.index.name = "datetime"
            regime_df["ticker"] = tk
            regime_df = regime_df.reset_index().set_index(["datetime", "ticker"])
            all_regime.append(regime_df)

        if not all_regime:
            return multi_asset

        regime_concat = pd.concat(all_regime)

        if "ichimoku_regime" in multi_asset.columns:
            multi_asset = multi_asset.drop(columns=["ichimoku_regime"])

        multi_asset = multi_asset.join(regime_concat, how="left")
        multi_asset["ichimoku_regime"] = (
            multi_asset["ichimoku_regime"].fillna(0).astype(int)
        )
        return multi_asset

    def add_model_predictions(
        self,
        multi_asset: pd.DataFrame,
        model: BaseEstimator,
        features: list[str],
        prefix: str = "primary",
        filter_prediction: int | None = None,
    ) -> pd.DataFrame:
        if multi_asset is None:
            raise ValueError("multi_asset is not initialised.")

        X = multi_asset[features]
        preds = model.predict(X)
        probas = model.predict_proba(X)

        prob_entropy = entropy(probas, axis=1)

        new_columns = pd.DataFrame(
            {f"{prefix}_pred": preds, f"{prefix}_entropy": prob_entropy},
            index=multi_asset.index,
        )

        proba_df = pd.DataFrame(
            probas,
            columns=[f"{prefix}_proba{i}" for i in range(probas.shape[1])],
            index=multi_asset.index,
        )

        new_columns = pd.concat([new_columns, proba_df], axis=1)

        multi_asset = pd.concat([multi_asset, new_columns], axis=1)
        if filter_prediction is not None:
            multi_asset = multi_asset[
                multi_asset[f"{prefix}_pred"] == filter_prediction
            ]

        return multi_asset

    def get_classifierengine_payload(
        self,
        multi_asset_train: pd.DataFrame,
        multi_asset_test: pd.DataFrame,
        features: list[str],
        tickers: list[str] | None = None,
    ) -> dict[str, pd.DataFrame | list[str] | str | None]:
        if multi_asset_train is None or multi_asset_test is None:
            raise ValueError("Training and test sets are not initialised.")

        features_copy = list(features)
        if self.weight_col and self.weight_col in features_copy:
            features_copy.remove(self.weight_col)

        X_train = multi_asset_train
        X_test = multi_asset_test

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


class AssetOrganiser:
    """
    Organises and prepares multi-asset data for a quantitative classifier.

    This class handles the conversion of a dictionary of disparate asset DataFrames
    into an aligned multi-index DataFrame, splits it based on a cutoff date,
    and manages the fitting and transformation process using a specified classifier.
    """

    def __init__(
        self,
        data_map: dict[str, pd.DataFrame] | None = None,
        cutoff_date: str | None = None,
        target_features: list[str] | None = None,
        weight_col: str | None = None,
        multi_asset: pd.DataFrame | None = None,
        label_factory: BaseLabelFactory | None = None,
    ) -> None:
        """
        Initialises the AssetOrganiser.

        Args:
            data_map (Optional[Dict[str, pd.DataFrame]]): Dictionary mapping tickers to
                their respective DataFrames.
            cutoff_date (str): The date string (e.g., 'YYYY-MM-DD') separating
                train and test sets.
            target_features (List[str]): List of column names to be used as targets (y).
            weight_col (Optional[str]): Column name in the DataFrame containing
                target weights. Defaults to ``"weight"`` when not supplied.
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

        self.data_map: dict[str, pd.DataFrame] | None = data_map
        self.cutoff_date: str = cutoff_date
        self.target_features: list[str] = list(target_features)
        self._weight_col: str = weight_col or "weight"
        self._label_factory: BaseLabelFactory | None = label_factory
        self.cusum_events_map: dict[str, pd.DatetimeIndex] | None = None

        self._panel = PanelBuilder(cutoff_date=self.cutoff_date)
        self._events = EventFilter()
        self._labels = LabelPipeline(
            label_factory=self._label_factory,
            weight_col=self._weight_col,
            target_features=self.target_features,
        )

        self.multi_asset: pd.DataFrame | None = multi_asset
        self.multi_asset_train: pd.DataFrame | None = None
        self.multi_asset_test: pd.DataFrame | None = None

        if self.multi_asset is not None:
            self._split_train_test()

    @property
    def label_factory(self) -> "BaseLabelFactory | None":
        return self._label_factory

    @label_factory.setter
    def label_factory(self, value: "BaseLabelFactory | None") -> None:
        self._label_factory = value
        self._labels.label_factory = value

    @property
    def weight_col(self) -> str:
        return self._weight_col

    @weight_col.setter
    def weight_col(self, value: str) -> None:
        self._weight_col = value
        self._labels.weight_col = value

    def _split_train_test(self) -> None:
        self.multi_asset_train, self.multi_asset_test = self._panel.split_train_test(
            self.multi_asset
        )

    def prepare_multi_asset_frame(self) -> None:
        if self.data_map is not None:
            self.multi_asset = self._panel.prepare(self.data_map, self.multi_asset)
        self._split_train_test()

    def downsample_to_events(
        self, events: pd.DatetimeIndex | list | set | dict[str, pd.DatetimeIndex]
    ) -> None:
        """
        Down-samples the multi-asset DataFrame to keep only the dates matching
        the specified events for each ticker.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()
        self.multi_asset = self._events.filter_to_events(self.multi_asset, events)
        self._split_train_test()

    def downsample_to_cusum_events(
        self,
        target_events_train: int | dict[str, int],
        filter_col: str,
        vol_col: str | None = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
        objective: str = "budget",
        t1_col: str | None = None,
    ) -> dict[str, float]:
        """
        Calibrates optimal alpha scalars on the training set and down-samples the
        multi-asset DataFrame using causal dynamic thresholds.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()
        filtered_multi_asset, events_map, calibrated_alphas = self._events.cusum_filter(
            self.multi_asset,
            self.multi_asset_train,
            target_events_train,
            filter_col,
            vol_col,
            span,
            alpha_min,
            alpha_max,
            alpha_step,
            objective,
            t1_col,
        )
        self.multi_asset = filtered_multi_asset
        self.cusum_events_map = events_map
        self._split_train_test()
        return calibrated_alphas

    def apply_continuous_labels(self, price_col: str = "Close") -> None:
        """
        Applies the label_factory strictly on the continuous, un-sampled price series.
        Injects the resulting 'label' and 't1' columns into the multi_asset DataFrame.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()
        self.multi_asset = self._labels.apply_continuous_labels(
            self.multi_asset, price_col
        )
        self._split_train_test()

    def apply_sample_weights(self, price_col: str = "Close") -> None:
        """
        Calculates sample weights strictly on the currently filtered multi_asset DataFrame.
        This must be run AFTER down-sampling (e.g., CUSUM) to correctly calculate concurrency.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()
        self.multi_asset = self._labels.apply_sample_weights(
            self.multi_asset, price_col
        )
        self._split_train_test()

    def build_learning_pipeline(
        self,
        target_events_train: int | dict[str, int],
        filter_col: str,
        price_col: str = "Close",
        vol_col: str | None = None,
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
        objective: str = "budget",
        t1_col: str | None = None,
    ) -> dict[str, float]:
        """
        Orchestrates the preparation pipeline to strictly prevent sequential data hazards.
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
        features: list[str],
        prefix: str = "primary",
        filter_prediction: int | None = None,
    ) -> None:
        """
        Fits the model on the multiasset data.
        Generates predictions and probability entropy from the provided model,
        injects them into the multi_asset DataFrame, and optionally filters the dataset.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()
        self.multi_asset = self._labels.add_model_predictions(
            self.multi_asset, model, features, prefix, filter_prediction
        )
        self._split_train_test()

    def get_classifierengine_payload(
        self,
        features: list[str],
        tickers: list[str] | None = None,
    ) -> dict[str, pd.DataFrame | list[str] | str | None]:
        """
        Extracts the prepared data and metadata into a dictionary suitable for
        unpacking (**kwargs) directly into `ClassifierEngine.run_pipeline`.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            self.prepare_multi_asset_frame()
        return self._labels.get_classifierengine_payload(
            self.multi_asset_train, self.multi_asset_test, features, tickers
        )

    def to_tsfeatures_format(
        self,
        value_col: str,
        subset: str = "all",
    ) -> pd.DataFrame:
        """
        Transforms the multi-asset DataFrame into the format required by Nixtla's `tsfeatures`.
        """
        if self.multi_asset is None and self.data_map is not None:
            self.prepare_multi_asset_frame()
        return self._panel.to_tsfeatures_format(
            self.multi_asset,
            self.multi_asset_train,
            self.multi_asset_test,
            value_col,
            subset,
        )

    def apply_ichimoku_regime(
        self, mode: str = "standard", displacement: int = 26
    ) -> None:
        """
        Computes the Ichimoku Cloud and injects a binary ``ichimoku_regime``
        column into ``self.multi_asset``, grouped by ticker.
        """
        if self.multi_asset is None:
            self.prepare_multi_asset_frame()
        self.multi_asset = self._labels.apply_ichimoku_regime(
            self.multi_asset, mode, displacement
        )
        self._split_train_test()

    def update_multi_asset(self, df: pd.DataFrame) -> None:
        """
        Overwrites the internal multi_asset panel dataset with engineered features
        and automatically re-synchronises the train and test split boundaries.
        """
        self.multi_asset = self._panel.update(df)
        self._split_train_test()

    def replace_features(
        self, transformed_df: pd.DataFrame, original_features: list[str]
    ) -> None:
        """
        Replaces the original features in the multi_asset panel dataset with
        transformed features, drops features that failed the pruning step,
        aligns the dataset to the transformed dataset's index (removing rows
        dropped during transformation), and re-synchronises the train/test split boundaries.
        """
        self.multi_asset = self._panel.replace_features(
            self.multi_asset, transformed_df, original_features
        )
        self._split_train_test()
