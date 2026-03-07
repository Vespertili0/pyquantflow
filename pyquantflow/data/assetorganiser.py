import pandas as pd
from typing import Dict, List, Optional
from ..model.classifier import BaseQuantClassifier
from .utils import align_and_ffill_multiasset, restructure_map_2_multiasset_df
from .schemas import validate_multi_asset


class AssetOrganiser:
    """
    Organises and prepares multi-asset data for a quantitative classifier.
    
    This class handles the conversion of a dictionary of disparate asset DataFrames
    into an aligned multi-index DataFrame, splits it based on a cutoff date, 
    and manages the fitting and transformation process using a specified classifier.
    """
    def __init__(self, classifier: BaseQuantClassifier, data_map: Dict[str, pd.DataFrame], 
                 cutoff_date: str, target_features: List[str], 
                 sample_weights: Optional[List[float]] = None) -> None:
        """
        Initializes the AssetOrganiser.

        Args:
            classifier (BaseQuantClassifier): The model pipeline to fit and transform the data.
            data_map (Dict[str, pd.DataFrame]): Dictionary mapping tickers to their respective DataFrames.
            cutoff_date (str): The date string (e.g., 'YYYY-MM-DD') separating train and test sets.
            target_features (List[str]): List of column names to be used as targets (y).
            sample_weights (Optional[List[float]]): Optional sample weights for the training data.
        """
        self.classifier: BaseQuantClassifier = classifier
        self.data_map: Dict[str, pd.DataFrame] = data_map
        self.cutoff_date: str = cutoff_date
        self.target_features: List[str] = target_features
        
        self.multi_asset: Optional[pd.DataFrame] = None
        self.multi_asset_train: Optional[pd.DataFrame] = None
        self.multi_asset_test: Optional[pd.DataFrame] = None
        self.multi_asset_transformed_test: Optional[pd.DataFrame] = None
        self.sample_weights: Optional[List[float]] = sample_weights

    def fit_classifier(self) -> None:
        """
        Fits the underlying classifier on the training set and transforms the test set.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            raise ValueError("Data not prepared. Call prepare_multi_asset_frame() first.")
            
        self.classifier.fit(
            X=self.multi_asset_train,
            y=self.multi_asset_train[self.target_features],
            sample_weight=self.sample_weights
        )
        self.multi_asset_transformed_test = self.classifier.transform(self.multi_asset_test)

    def prepare_multi_asset_frame(self) -> None:
        """
        converts data_map to Date-Ticker multi-index dataframe
        """
        self.multi_asset = validate_multi_asset(
            align_and_ffill_multiasset(
                restructure_map_2_multiasset_df(self.data_map)
            )
        )
        self.multi_asset_train = self.multi_asset[
            self.multi_asset.index.get_level_values("datetime") < self.cutoff_date
        ]
        self.multi_asset_test  = self.multi_asset[
            self.multi_asset.index.get_level_values("datetime") >= self.cutoff_date
        ]
        return None
    
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
        return self.multi_asset_transformed_test.xs(ticker, level='ticker')