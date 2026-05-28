import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from ..model.classifier import BaseQuantClassifier
from .utils import align_and_ffill_multiasset, restructure_map_2_multiasset_df
from .labels import get_cusum_events, calibrate_cusum_alpha


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
        classifier: Optional[BaseQuantClassifier] = None,
        multi_asset: Optional[pd.DataFrame] = None,
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
            classifier (Optional[BaseQuantClassifier]): Optional model pipeline to fit
                and transform the data.
            multi_asset (Optional[pd.DataFrame]): Pre-constructed multi-asset DataFrame.
        """
        if data_map is None and multi_asset is None:
            raise ValueError("Either 'data_map' or 'multi_asset' must be provided.")
        if data_map is not None and multi_asset is not None:
            raise ValueError("Cannot provide both 'data_map' and 'multi_asset'.")
        if cutoff_date is None:
            raise ValueError("'cutoff_date' is required.")
        if target_features is None:
            raise ValueError("'target_features' is required.")

        self.classifier: Optional[BaseQuantClassifier] = classifier
        self.data_map: Optional[Dict[str, pd.DataFrame]] = data_map
        self.cutoff_date: str = cutoff_date
        self.target_features: List[str] = target_features
        self.weight_col: Optional[str] = weight_col

        self.multi_asset: Optional[pd.DataFrame] = multi_asset
        self.multi_asset_train: Optional[pd.DataFrame] = None
        self.multi_asset_test: Optional[pd.DataFrame] = None
        self.multi_asset_transformed_test: Optional[pd.DataFrame] = None

        if self.multi_asset is not None:
            self._split_train_test()

    def _split_train_test(self) -> None:
        """
        Splits the multi_asset DataFrame into train and test sets
        based on the cutoff date.
        """
        self.multi_asset_train = self.multi_asset[
            self.multi_asset.index.get_level_values("datetime") < self.cutoff_date
        ]
        self.multi_asset_test = self.multi_asset[
            self.multi_asset.index.get_level_values("datetime") >= self.cutoff_date
        ]

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
        price_col: str = "close",
        span: int = 100,
        alpha_min: float = 0.5,
        alpha_max: float = 3.0,
        alpha_step: float = 0.1,
    ) -> Dict[str, float]:
        """
        Calibrates optimal alpha scalars on the training set (Event Budgeting)
        and down-samples the multi-asset DataFrame using causal dynamic thresholds.

        Parameters
        ----------
        target_events_train : int | Dict[str, int]
            The target event count for the training fold. If int, applied to all tickers.
            If dict, maps ticker to specific target count.
        price_col : str, default='close'
            The name of the price column in the DataFrame to run CUSUM on.
        span : int, default=100
            The EWMA span for calculating dynamic volatility.
        alpha_min : float, default=0.5
            Minimum alpha multiplier.
        alpha_max : float, default=3.0
            Maximum alpha multiplier.
        alpha_step : float, default=0.1
            Grid search step size.

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

            # Extract training prices for ticker
            try:
                ticker_train_prices = self.multi_asset_train.xs(tk, level="ticker")[
                    price_col
                ]
            except KeyError:
                # If ticker has no data in training fold, use default alpha
                calibrated_alphas[tk] = alpha_min
                continue

            # Run calibration strictly on training set prices
            alpha = calibrate_cusum_alpha(
                prices=ticker_train_prices,
                target_events=target,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                alpha_step=alpha_step,
                span=span,
            )
            calibrated_alphas[tk] = alpha

        # 2. Run CUSUM filter with dynamic, volatility-adjusted threshold
        # using the calibrated/frozen alphas across the ENTIRE dataset (no leakage)
        events_map = {}
        for tk in tickers:
            alpha = calibrated_alphas[tk]
            prices_all = self.multi_asset.xs(tk, level="ticker")[price_col]

            # Calculate causal EWMA volatility on entire price series
            returns_all = prices_all.pct_change()
            vol_all = returns_all.ewm(span=span).std()
            threshold_all = alpha * vol_all

            # Filter events
            events = get_cusum_events(prices_all, threshold_all)
            events_map[tk] = events

        # 3. Down-sample the organiser's multi-asset DataFrame using these events
        self.downsample_to_events(events_map)

        return calibrated_alphas

    def fit_quant_classifier(self) -> None:
        """
        Fits the underlying classifier on the training set and transforms the test set.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            raise ValueError(
                "Data not prepared. Call prepare_multi_asset_frame() first."
            )

        if self.classifier is None:
            raise ValueError("No classifier was provided during initialization.")

        # Optional: Extract sample weights if weight_col is specified
        sw = None
        if self.weight_col and self.weight_col in self.multi_asset_train.columns:
            sw = self.multi_asset_train[self.weight_col].values

        self.classifier.fit(
            X=self.multi_asset_train,
            y=self.multi_asset_train[self.target_features],
            sample_weight=sw,
        )
        self.transform_test_set()

        return None

    def transform_test_set(self) -> None:
        """
        Predict-transforms the test set using the fitted classifier.
        """
        self.multi_asset_transformed_test = self.classifier.transform(
            self.multi_asset_test
        )

        return None

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
        self, features: List[str]
    ) -> Dict[str, pd.DataFrame | List[str] | str | None]:
        """
        Extracts the prepared data and metadata into a dictionary suitable for
        unpacking (**kwargs) directly into `ClassifierEngine.run_pipeline`.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            self.prepare_multi_asset_frame()

        # Remove weight_col from features list to strictly separate metadata
        if self.weight_col and self.weight_col in features:
            features.remove(self.weight_col)

        return {
            "X_train": self.multi_asset_train,
            "y_train": self.multi_asset_train[self.target_features],
            "X_test": self.multi_asset_test,
            "y_test": self.multi_asset_test[self.target_features],
            "features": features,
            "weight_col": self.weight_col,
        }

    def get_transformed_multiasset_testdata(self) -> pd.DataFrame:
        """
        Returns the transformed test data containing predictions.

        Returns:
            pd.DataFrame: Transformed multi-asset test DataFrame.
        """
        if self.multi_asset_transformed_test is None:
            raise ValueError("Test data not transformed. Fit the classifier first.")
        return self.multi_asset_transformed_test

    def get_transformed_test_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves the transformed test data for a specific ticker.

        Args:
            ticker (str): The symbol/ticker to retrieve.

        Returns:
            pd.DataFrame: Transformed test DataFrame for the given ticker.
        """
        if self.multi_asset_transformed_test is None:
            raise ValueError("Test data not transformed. Fit the classifier first.")
        return self.multi_asset_transformed_test.xs(ticker, level="ticker")
