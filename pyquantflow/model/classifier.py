from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# from sklearn.utils.validation import check_is_fitted
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone


class BaseQuantClassifier(ABC, BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Abstract Base Class for quantitative classifiers used in the pipeline.
    Ensures that any custom classifier implements the necessary scikit-learn
    compatible interfaces, predicting probabilities and transforming data.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "BaseQuantClassifier":
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
    def __init__(
        self,
        primary_model,
        secondary_model,
        primary_features,
        secondary_features,
        cv_generator=None,
        prefitted=True,
        proba_class_idx: int = 1,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.primary_features = primary_features
        self.secondary_features = secondary_features
        self.cv_generator = cv_generator
        self.prefitted = prefitted
        self.proba_class_idx = proba_class_idx

        if self.prefitted:
            self.primary_model_ = self.primary_model
            self.secondary_model_ = self.secondary_model
        else:
            self.primary_model_ = None
            self.secondary_model_ = None

    def _calculate_entropy(self, probas):
        # Calculate Shannon Entropy: H = -sum(p * log(p))
        return entropy(probas, axis=1).reshape(-1, 1)

    def fit(self, X, y, sample_weight=None):
        self.prefitted = False
        self.primary_model_ = clone(self.primary_model)
        self.secondary_model_ = clone(self.secondary_model)

        y_primary = y.iloc[:, 0] if hasattr(y, "iloc") else y[:, 0]
        y_secondary = y.iloc[:, 1] if hasattr(y, "iloc") else y[:, 1]

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
                fold_primary.fit(
                    X.iloc[train_idx][self.primary_features],
                    y_primary.iloc[train_idx]
                    if hasattr(y_primary, "iloc")
                    else y_primary[train_idx],
                    sample_weight=fold_sw,
                )
            else:
                fold_primary.fit(
                    X.iloc[train_idx][self.primary_features],
                    y_primary.iloc[train_idx]
                    if hasattr(y_primary, "iloc")
                    else y_primary[train_idx],
                )

            # Predict on validation fold
            fold_probas = fold_primary.predict_proba(
                X.iloc[val_idx][self.primary_features]
            )
            oof_entropy[val_idx] = self._calculate_entropy(fold_probas)

        # Handle potential gaps from purging/embargoing
        if np.isnan(oof_entropy).any():
            oof_entropy = pd.DataFrame(oof_entropy).ffill().bfill().values

        # 2. Final Fits on all data
        if sw is not None:
            self.primary_model_.fit(
                X[self.primary_features], y_primary, sample_weight=sw
            )
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

        For binary models ``primary_proba`` and ``secondary_proba`` contain
        the scalar probability for the positive class (``self.proba_class_idx``,
        default 1). For multi-class models those columns still contain the
        focal-class scalar, and additional per-class columns are added:
        ``primary_proba_0``, ``primary_proba_1``, ... (and equivalent secondary
        columns) for downstream inspection.
        """
        # check_is_fitted(self)
        X_out = X.copy()

        # Primary outputs
        X_out["primary_pred"] = self.primary_model_.predict(X[self.primary_features])
        probas = self.primary_model_.predict_proba(X[self.primary_features])
        X_out["primary_proba"] = probas[:, self.proba_class_idx]
        if probas.shape[1] > 2:
            for i in range(probas.shape[1]):
                X_out[f"primary_proba_{i}"] = probas[:, i]
        X_out["primary_entropy"] = self._calculate_entropy(probas)

        # Prepare secondary inputs
        X_secondary = np.hstack(
            [
                X[self.secondary_features].values,
                X_out["primary_entropy"].values.reshape(-1, 1),
            ]
        )

        # Secondary outputs
        sec_probas = self.secondary_model_.predict_proba(X_secondary)
        X_out["secondary_proba"] = sec_probas[:, self.proba_class_idx]
        if sec_probas.shape[1] > 2:
            for i in range(sec_probas.shape[1]):
                X_out[f"secondary_proba_{i}"] = sec_probas[:, i]
        X_out["final_decision"] = self.secondary_model_.predict(X_secondary)

        return X_out

    def predict(self, X):
        # check_is_fitted(self)
        probas = self.primary_model_.predict_proba(X[self.primary_features])
        proba_entropy = self._calculate_entropy(probas)
        X_secondary = np.hstack([X[self.secondary_features].values, proba_entropy])
        return self.secondary_model_.predict(X_secondary)

    def predict_proba(self, X):
        # check_is_fitted(self)
        probas = self.primary_model_.predict_proba(X[self.primary_features])
        proba_entropy = self._calculate_entropy(probas)
        X_secondary = np.hstack([X[self.secondary_features].values, proba_entropy])
        return self.secondary_model_.predict_proba(X_secondary)


class IchimokuBaselineClassifier(BaseEstimator, ClassifierMixin):
    """
    A stateless, rules-based baseline classifier driven by the pre-computed
    ``ichimoku_regime`` column in the feature matrix.

    Implements the scikit-learn estimator API so it can be evaluated inside
    ``StrategyLab`` cross-validation loops exactly like an ML model. The regime
    signal must already exist as an integer column (0 or 1) in ``X`` before any
    call to ``predict`` or ``predict_proba`` — typically injected upstream by
    ``AssetOrganiser.apply_ichimoku_regime()``.

    Parameters
    ----------
    regime_col : str, default "ichimoku_regime"
        Name of the binary regime column in the input DataFrame.
    """

    def __init__(self, regime_col: str | int = "ichimoku_regime") -> None:
        self.regime_col = regime_col

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        sample_weight=None,
    ) -> "IchimokuBaselineClassifier":
        """
        No-op fit. The classifier is entirely rule-based and requires no
        training. Stores ``classes_`` to satisfy sklearn validators.

        When ``y`` is supplied, ``classes_`` is inferred from the unique
        values in ``y`` so that multi-class label sets (e.g. ``{-1, 0, 1}``
        or ``{0, 1, 2}``) are reflected correctly. If ``y`` is ``None`` the
        default binary set ``[0, 1]`` is used as a safe fallback.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must contain ``self.regime_col``.
        y : array-like of shape (n_samples,) or None, default None
            Target labels used solely to infer ``classes_``. The classifier
            does not use ``y`` during inference.
        sample_weight : ignored

        Returns
        -------
        self
        """
        if y is not None:
            self.classes_ = np.unique(np.asarray(y).ravel())
        else:
            self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Returns the ``ichimoku_regime`` column as an integer prediction array.

        Supports both :class:`pandas.DataFrame` (preferred) and plain
        :class:`numpy.ndarray` inputs so the classifier is compatible with
        scikit-learn pipelines that do not propagate column names (i.e. where
        ``set_output(transform='pandas')`` is not globally enforced).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.  When a DataFrame is supplied the regime signal is
            extracted by column name (``self.regime_col``).  When a NumPy array
            is supplied, ``self.regime_col`` is used as a positional column
            index if it is already an ``int``; otherwise column index ``0`` is
            assumed, matching the convention that ``AssetOrganiser`` injects the
            regime column as the sole or first feature.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.

        Raises
        ------
        TypeError
            If ``X`` is neither a :class:`pandas.DataFrame` nor a
            :class:`numpy.ndarray`.
        KeyError
            If ``self.regime_col`` is not present in a DataFrame ``X``.
        ValueError
            If ``X`` is a NumPy array whose column count is too small for the
            requested positional index.
        """
        if isinstance(X, np.ndarray):
            # Fallback for pipelines that strip column names.
            # Use regime_col directly if it is an integer index; otherwise
            # fall back to position 0 (AssetOrganiser convention).
            col_idx = self.regime_col if isinstance(self.regime_col, int) else 0
            if X.ndim == 1:
                return X.astype(int)
            if X.shape[1] <= col_idx:
                raise ValueError(
                    f"IchimokuBaselineClassifier: NumPy array has {X.shape[1]} "
                    f"column(s) but regime column index {col_idx} was requested. "
                    "Ensure the feature matrix is constructed with the regime "
                    "column at the expected position."
                )
            return X[:, col_idx].astype(int)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"IchimokuBaselineClassifier expects a pd.DataFrame or "
                f"np.ndarray. Received {type(X).__name__}."
            )
        if self.regime_col not in X.columns:
            raise KeyError(
                f"Column '{self.regime_col}' not found in X. "
                "Ensure AssetOrganiser.apply_ichimoku_regime() has been called."
            )
        return X[self.regime_col].to_numpy(dtype=int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Returns a probability matrix consistent with the regime signal.

        For binary classification (``classes_ == [0, 1]``), each row is
        ``[1 - regime, regime]``.  For multi-class targets the returned
        matrix has shape ``(n_samples, n_classes)`` where column ``i``
        contains 1.0 for samples whose predicted class equals
        ``self.classes_[i]`` and 0.0 otherwise, preserving the hard
        threshold-free decision contract.

        Accepts both :class:`pandas.DataFrame` and :class:`numpy.ndarray`
        inputs; all input-handling logic is delegated to :meth:`predict`.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix containing the regime signal.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, n_classes)`` where each row sums to 1.
        """
        classes = getattr(self, "classes_", np.array([0, 1]))
        preds = self.predict(X)
        if len(classes) == 2 and np.array_equal(classes, np.array([0, 1])):
            regime = preds.astype(float)
            return np.column_stack([1.0 - regime, regime])
        return np.column_stack(
            [(preds == c).astype(float) for c in classes]
        )
