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
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray], 
        model_factory: Callable[[optuna.Trial], Any], 
        cv: BaseCrossValidator, 
        metric: Callable = f1_score,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric_kwargs: Optional[dict] = None
    ) -> optuna.Study:
        """
        Executes the optimization loop using the existing self.study attribute.
        
        Args:
            X, y: Training data (will be converted to numpy internally).
            model_factory: Function taking a 'trial' returning an un-fitted sklearn estimator.
            cv: Sklearn-compatible CV splitter.
            metric: Scoring function (y_true, y_pred) -> float.
        """
        metric_kwargs = metric_kwargs or {}

        # Define optuna objective function
        def objective(trial: optuna.Trial):
            model = model_factory(trial)
            fold_scores = []

            # Loop over CV splits using the Numpy arrays
            for step, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                
                # Slicing numpy arrays directly (no .iloc needed)
                X_train_fold = X.iloc[train_idx]
                X_val_fold   = X.iloc[val_idx]

                y_train_fold = y.iloc[train_idx]
                y_val_fold   = y.iloc[val_idx]


                model.fit(X_train_fold, y_train_fold)

                # Predict logic
                if metric.__name__ in ("log_loss", "roc_auc_score"):
                    # For metrics requiring probabilities
                    proba = model.predict_proba(X_val_fold)
                    # Handle binary classification case (return prob of positive class)
                    preds = proba[:, 1] if proba.shape[1] == 2 else proba
                else:
                    # For metrics requiring class labels
                    preds = model.predict(X_val_fold)

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