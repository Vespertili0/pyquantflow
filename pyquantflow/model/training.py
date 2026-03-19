import optuna
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import f1_score



class HyperparameterOptimiser:
    """
    A factory class to manage Optuna studies for Sklearn model development.
    Supports stopping early (Pruning) if cross-validation folds perform poorly.
    """
    def __init__(
        self, 
        study_name: str, 
        storage_uri: str = None, 
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None
    ):
        """
        Args:
            study_name: Unique identifier for the study.
            storage_uri: SQL connection string for persistence.
            direction: 'maximize' or 'minimize'.
        """
        self.study_name = study_name
        self.storage_uri = storage_uri
        self.direction = direction
        self.sampler = sampler or optuna.samplers.TPESampler(seed=42)
        
        # Initialize the study as a class attribute immediately
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_uri,
            direction=self.direction,
            load_if_exists=True,
            sampler=self.sampler
        )

    def run(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray, pd.DataFrame], 
        features: list[str],
        model_factory: Callable[[optuna.Trial], Any], 
        cv: BaseCrossValidator, 
        weight_col: Optional[str] = None,
        t1_col: Optional[str] = None,
        metric: Callable = f1_score,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric_kwargs: Optional[dict] = None
    ) -> optuna.Study:
        """
        Executes the optimization loop using the existing self.study attribute.
        
        Args:
            X, y: Training data (X must be a pandas DataFrame containing features & metadata).
            features: List of column names in X to strictly use as training features.
            model_factory: Function taking a 'trial' returning an un-fitted sklearn estimator.
            cv: Sklearn-compatible CV splitter.
            weight_col: Optional column name for sample weights.
            t1_col: Optional column name for event end times (purging metadata).
            metric: Scoring function (y_true, y_pred) -> float.
        """
        metric_kwargs = metric_kwargs or {}

        # Define optuna objective function
        def objective(trial: optuna.Trial):
            model = model_factory(trial)
            fold_scores = []

            # Loop over CV splits using the Numpy arrays
            for step, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                
                # Slice entire rows to capture metadata
                X_train_fold = X.iloc[train_idx]
                X_val_fold   = X.iloc[val_idx]

                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_val_fold   = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

                fit_params = {}
                if weight_col and weight_col in X_train_fold.columns:
                    sample_weight = X_train_fold[weight_col].values
                    # Extract final step name if pipeline, else just use sample_weight
                    if hasattr(model, "steps"):
                        final_step_name = model.steps[-1][0]
                        fit_params[f"{final_step_name}__sample_weight"] = sample_weight
                    else:
                        fit_params["sample_weight"] = sample_weight

                # Isolate pure features for fitting and predicting
                X_train_features = X_train_fold[features]
                X_val_features   = X_val_fold[features]

                model.fit(X_train_features, y_train_fold, **fit_params)

                # Predict logic
                if metric.__name__ in ("log_loss", "roc_auc_score"):
                    # For metrics requiring probabilities
                    proba = model.predict_proba(X_val_features)
                    # Handle binary classification case (return prob of positive class)
                    preds = proba[:, 1] if proba.shape[1] == 2 else proba
                else:
                    # For metrics requiring class labels
                    preds = model.predict(X_val_features)

                # Score
                score = metric(y_val_fold, preds, **metric_kwargs)
                fold_scores.append(score)

                # Report & Prune based on mean of current folds (or simply current fold)
                # Reporting the running average gives a smoother pruning curve
                trial.report(np.mean(fold_scores), step=step)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return np.mean(fold_scores)

        # Optimize the existing study object
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        return self.study