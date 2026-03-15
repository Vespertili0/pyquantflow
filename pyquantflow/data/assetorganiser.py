import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import entropy
from sklearn.base import BaseEstimator
from ..model.classifier import BaseQuantClassifier
from .utils import align_and_ffill_multiasset, restructure_map_2_multiasset_df


class AssetOrganiser:
    """
    Organises and prepares multi-asset data for a quantitative classifier.
    
    This class handles the conversion of a dictionary of disparate asset DataFrames
    into an aligned multi-index DataFrame, splits it based on a cutoff date, 
    and manages the fitting and transformation process using a specified classifier.
    """
    def __init__(self, data_map: Dict[str, pd.DataFrame], 
                 cutoff_date: str, target_features: List[str], 
                 weight_col: Optional[str] = None,
                 classifier: Optional[BaseQuantClassifier] = None) -> None:
        """
        Initializes the AssetOrganiser.

        Args:
            data_map (Dict[str, pd.DataFrame]): Dictionary mapping tickers to their respective DataFrames.
            cutoff_date (str): The date string (e.g., 'YYYY-MM-DD') separating train and test sets.
            target_features (List[str]): List of column names to be used as targets (y).
            weight_col (Optional[str]): Optional column name in the DataFrame containing target weights.
            classifier (Optional[BaseQuantClassifier]): Optional model pipeline to fit and transform the data.
        """
        self.classifier: Optional[BaseQuantClassifier] = classifier
        self.data_map: Dict[str, pd.DataFrame] = data_map
        self.cutoff_date: str = cutoff_date
        self.target_features: List[str] = target_features
        self.weight_col: Optional[str] = weight_col
        
        self.multi_asset: Optional[pd.DataFrame] = None
        self.multi_asset_train: Optional[pd.DataFrame] = None
        self.multi_asset_test: Optional[pd.DataFrame] = None
        self.multi_asset_transformed_test: Optional[pd.DataFrame] = None

    def _split_train_test(self) -> None:
        """
        Splits the multi_asset DataFrame into train and test sets based on the cutoff date.
        """
        self.multi_asset_train = self.multi_asset[
            self.multi_asset.index.get_level_values("datetime") < self.cutoff_date
        ]
        self.multi_asset_test  = self.multi_asset[
            self.multi_asset.index.get_level_values("datetime") >= self.cutoff_date
        ]        

    def prepare_multi_asset_frame(self) -> None:
        """
        converts data_map to Date-Ticker multi-index dataframe
        """
        self.multi_asset = align_and_ffill_multiasset(
            restructure_map_2_multiasset_df(self.data_map)
        )
        self._split_train_test()

        return None

    def fit_quant_classifier(self) -> None:
        """
        Fits the underlying classifier on the training set and transforms the test set.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            raise ValueError("Data not prepared. Call prepare_multi_asset_frame() first.")
        
        if self.classifier is None:
            raise ValueError("No classifier was provided during initialization.")
            
        # Optional: Extract sample weights if weight_col is specified
        sw = None
        if self.weight_col and self.weight_col in self.multi_asset_train.columns:
            sw = self.multi_asset_train[self.weight_col].values
            
        self.classifier.fit(
            X=self.multi_asset_train,
            y=self.multi_asset_train[self.target_features],
            sample_weight=sw
        )
        self.multi_asset_transformed_test = self.classifier.transform(self.multi_asset_test)

    def add_model_predictions(self, model: BaseEstimator, features: List[str], filter_prediction: Optional[int] = None) -> None:
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
        new_columns = pd.DataFrame({
            "primary_pred": preds,
            # Assuming binary classification where class 1 is the positive outcome
            "primary_proba": probas[:, 1] if probas.shape[1] > 1 else probas[:, 0],
            "primary_entropy": prob_entropy
        }, index=self.multi_asset.index)
        
        self.multi_asset = pd.concat([self.multi_asset, new_columns], axis=1)
        
        # Apply Meta-Labelling filter (e.g. only keep trades the primary model took)
        if filter_prediction is None:
            pass
        else:
            self.multi_asset = self.multi_asset[self.multi_asset["primary_pred"] == filter_prediction]
            
        # Re-split the chronological train/test sets to reflect the new features and filtering
        self._split_train_test()

    def get_classifierengine_payload(self, features: List[str]) -> Dict[str, pd.DataFrame | List[str] | str | None]:
        """
        Extracts the prepared data and metadata into a dictionary suitable for 
        unpacking (**kwargs) directly into `ClassifierEngine.run_pipeline`.
        """
        if self.multi_asset_train is None or self.multi_asset_test is None:
            self.prepare_multi_asset_frame()            
        
        # Remove weight_col from features list if it exists to strictly separate metadata
        if self.weight_col and self.weight_col in features:
            features.remove(self.weight_col)

        return {
            "X_train": self.multi_asset_train[features],
            "y_train": self.multi_asset_train[self.target_features],
            "X_test": self.multi_asset_test[features],
            "y_test": self.multi_asset_test[self.target_features],
            "features": features,
            "weight_col": self.weight_col
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
        return self.multi_asset_transformed_test.xs(ticker, level='ticker')