import optuna
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Dict, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import f1_score
import logging

try:
    import mlflow
except ImportError:
    mlflow = None

from pyquantflow.model.training import HyperparameterOptimiser

logger = logging.getLogger(__name__)

class BaseModelEngine(ABC):
    """
    Abstract base class for model lifecycle management.
    Enforces a structure for validation and registration.
    """
    
    @abstractmethod
    def validate(
        self, 
        model: Any, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        metric: Callable
    ) -> Dict[str, float]:
        """
        Validate the model on a hold-out set.
        Should return a dictionary of metrics.
        """
        pass

    @abstractmethod
    def register(
        self, 
        model: Any, 
        params: Dict[str, Any], 
        metrics: Dict[str, float], 
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> None:
        """
        Register the model, parameters, and metrics to a tracking server (e.g. MLflow).
        """
        pass

class ModelOrchestrationEngine(BaseModelEngine):
    """
    Orchestrates the full ML pipeline:
    1. Hyperparameter Optimization (via HyperparameterOptimiser)
    2. Re-training best model on full training set
    3. Evaluation on hold-out test set
    4. Optional MLflow registration
    """

    def __init__(self, optimiser: HyperparameterOptimiser):
        self.optimiser = optimiser

    def validate(
        self, 
        model: Any, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        metric: Callable = f1_score,
        metric_kwargs: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Computes the metric score on X, y.
        """
        metric_kwargs = metric_kwargs or {}
        
        # Ensure model is fitted (assumed to be done by caller or prior step)
        # We try to handle proba if the metric suggests it, similar to training.py
        if hasattr(model, "predict_proba") and metric.__name__ in ("log_loss", "roc_auc_score"):
             preds = model.predict_proba(X)
             # Handle binary classification if needed
             if preds.ndim > 1 and preds.shape[1] == 2:
                 preds = preds[:, 1]
        else:
            preds = model.predict(X)
            
        score = metric(y, preds, **metric_kwargs)
        return {metric.__name__: score}

    def register(
        self, 
        model: Any, 
        params: Dict[str, Any], 
        metrics: Dict[str, float], 
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> None:
        """
        Registers to MLflow if available.
        """
        if mlflow is None:
            logger.warning("MLflow is not installed. Skipping registration.")
            print("MLflow is not installed. Skipping registration.")
            return

        # Set or create experiment
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            # We assume sklearn model for now based on context, but try generic first
            if hasattr(mlflow, "sklearn"):
                 mlflow.sklearn.log_model(model, "model")
            else:
                 # Fallback if specific flavor handling is desired
                 logger.info("Generic MLflow model logging not fully implemented, using sklearn default.")
                 mlflow.sklearn.log_model(model, "model")
            
            logger.info("Model registered to MLflow successfully.")
            print(f"Model registered to MLflow experiment '{experiment_name}' with run_name '{run_name}'.")

    def run_pipeline(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        model_factory: Callable[[optuna.Trial], Any],
        cv: BaseCrossValidator,
        metric: Callable = f1_score,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        metric_kwargs: Optional[dict] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> Any:
        """
        Executes the full pipeline.
        """
        # 1. Run Optimization
        print("Starting Hyperparameter Optimization...")
        study = self.optimiser.run(
            X=X_train,
            y=y_train,
            model_factory=model_factory,
            cv=cv,
            metric=metric,
            n_trials=n_trials,
            timeout=timeout,
            metric_kwargs=metric_kwargs
        )
        
        best_params = study.best_params
        best_value = study.best_value
        print(f"Optimization complete. Best CV Score: {best_value}")
        print(f"Best Params: {best_params}")

        # 2. Re-instantiate the best model using FixedTrial
        # This allows us to use the same logic in model_factory without modification
        print("Re-instantiating best model...")
        fixed_trial = optuna.trial.FixedTrial(best_params)
        best_model = model_factory(fixed_trial)

        # 3. Fit on ALL training data
        print("Retraining best model on full training set...")
        best_model.fit(X_train, y_train)

        # 4. Validate on Hold-out Test Set
        print("Validating on hold-out test set...")
        validation_metrics = self.validate(best_model, X_test, y_test, metric, metric_kwargs)
        print(f"Validation Metrics: {validation_metrics}")

        # 5. Register
        print("Attempting MLflow registration...")
        # Combine best params and extra info if needed
        self.register(
            model=best_model,
            params=best_params,
            metrics=validation_metrics,
            experiment_name=experiment_name,
            run_name=run_name
        )

        return best_model
