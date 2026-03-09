from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from scipy.stats import entropy
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseQuantClassifier(ABC, BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Abstract Base Class for quantitative classifiers used in the pipeline.
    Ensures that any custom classifier implements the necessary scikit-learn
    compatible interfaces, predicting probabilities and transforming data.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, sample_weight: np.ndarray | None = None) -> 'BaseQuantClassifier':
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass


class PrimarySecondaryClassifier(BaseQuantClassifier):
    def __init__(self, primary_model, secondary_model, primary_features, 
                 secondary_features, cv_generator=None):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.primary_features = primary_features
        self.secondary_features = secondary_features
        self.cv_generator = cv_generator

    def _calculate_entropy(self, probas):
        # Calculate Shannon Entropy: H = -sum(p * log(p))
        return entropy(probas, axis=1).reshape(-1, 1)

    def fit(self, X, y, sample_weight=None):
        self.primary_model_ = clone(self.primary_model)
        self.secondary_model_ = clone(self.secondary_model)
        
        y_primary = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
        y_secondary = y.iloc[:, 1] if hasattr(y, 'iloc') else y[:, 1]

        # Convert sample_weight to a numpy array for easy slicing during CV
        sw = np.asarray(sample_weight) if sample_weight is not None else None

        cv = self.cv_generator if self.cv_generator is not None else 5
        oof_entropy = np.full((len(X), 1), np.nan)

        # 1. Generate Out-of-Fold Entropy for the secondary model
        for train_idx, val_idx in cv.split(X, y_primary):
            fold_primary = clone(self.primary_model)
            
            # Slice sample weights if they exist
            fold_sw = sw[train_idx] if sw is not None else None
            
            # Fit on training fold
            if fold_sw is not None:
                fold_primary.fit(X.iloc[train_idx][self.primary_features], 
                                 y_primary.iloc[train_idx], sample_weight=fold_sw)
            else:
                fold_primary.fit(X.iloc[train_idx][self.primary_features], 
                                 y_primary.iloc[train_idx])
            
            # Predict on validation fold
            fold_probas = fold_primary.predict_proba(X.iloc[val_idx][self.primary_features])
            oof_entropy[val_idx] = self._calculate_entropy(fold_probas)

        # Handle potential gaps from purging/embargoing
        if np.isnan(oof_entropy).any():
            oof_entropy = pd.DataFrame(oof_entropy).ffill().bfill().values

        # 2. Final Fits on all data
        if sw is not None:
            self.primary_model_.fit(X[self.primary_features], y_primary, sample_weight=sw)
        else:
            self.primary_model_.fit(X[self.primary_features], y_primary)

        X_secondary_train = np.hstack([X[self.secondary_features].values, oof_entropy])
        
        if sw is not None:
            self.secondary_model_.fit(X_secondary_train, y_secondary, sample_weight=sw)
        else:
            self.secondary_model_.fit(X_secondary_train, y_secondary)

        return self

    def transform(self, X):
        """
        Enriches the input DataFrame with model predictions and probabilities.
        """
        check_is_fitted(self)
        X_out = X.copy()
        
        # Primary outputs
        X_out['primary_pred'] = self.primary_model_.predict(X[self.primary_features])
        probas = self.primary_model_.predict_proba(X[self.primary_features])
        X_out['primary_proba'] = probas[:, 1]
        X_out['primary_entropy'] = self._calculate_entropy(probas)
        
        # Prepare secondary inputs
        X_secondary = np.hstack([X[self.secondary_features].values, 
                                 X_out['primary_entropy'].values.reshape(-1, 1)])
        
        # Secondary outputs
        X_out['secondary_proba'] = self.secondary_model_.predict_proba(X_secondary)[:, 1]
        X_out['final_decision'] = self.secondary_model_.predict(X_secondary)
        
        return X_out

    def predict(self, X):
        check_is_fitted(self)
        probas = self.primary_model_.predict_proba(X[self.primary_features])
        proba_entropy = self._calculate_entropy(probas)
        X_secondary = np.hstack([X[self.secondary_features].values, proba_entropy])
        return self.secondary_model_.predict(X_secondary)

    def predict_proba(self, X):
        check_is_fitted(self)
        probas = self.primary_model_.predict_proba(X[self.primary_features])
        proba_entropy = self._calculate_entropy(probas)
        X_secondary = np.hstack([X[self.secondary_features].values, proba_entropy])
        return self.secondary_model_.predict_proba(X_secondary)