import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skfolio import Population, RatioMeasure, RiskMeasure, PerfMeasure
from typing import Any, Callable, Dict, List, Optional, Union
from skfolio.model_selection import (
    cross_val_predict, WalkForward, CombinatorialPurgedCV,
    optimal_folds_number, MultipleRandomizedCV
)


class StrategyLab:
    """
    A laboratory for evaluating and optimizing portfolio strategies.
    
    This class manages the lifecycle of strategy evaluation, including 
    hyperparameter search, cross-validation, and robustness testing.
    """
    def __init__(self, returns: pd.DataFrame, strategy_dict: Dict[str, Dict[str, Any]]):
        """
        Initialize the StrategyLab.

        Args:
            returns (pd.DataFrame): DataFrame of asset returns.
            strategy_dict (Dict[str, Dict[str, Any]]): Dictionary mapping strategy names 
                to their configurations (e.g., estimator and parameter grid).
        """
        self.returns = returns
        self.strategy_dict = strategy_dict
        self.best_estimators: Dict[str, Any] = {}
        self.cv = WalkForward(train_size=126, test_size=63)


    def search_strategy_hyperparameters(
        self, 
        scoring: Union[str, Callable], 
        cv: Optional[Any] = None
    ) -> None:
        """
        Perform grid search to find the best hyperparameters for each strategy.

        Args:
            scoring (Union[str, Callable]): Scoring metric for optimization.
            cv (Optional[Any]): Cross-validation strategy. Defaults to WalkForward.
            
        Returns:
            None: Fits the models and stores the best estimators in `self.best_estimators`.
        """
        if cv is None:
            cv = self.cv

        for name, config in self.strategy_dict.items():
            print(f"Optimizing {name}...")
            # 1. HYPERPARAMETER SEARCH ()
            grid_search = GridSearchCV(
                estimator=config['estimator'],
                param_grid=config['grid'],
                cv=cv,
                scoring=scoring,
                refit=True, # Automatically refit the best model on ALL data
                n_jobs=-1
            )
            grid_search.fit(self.returns)
            self.best_estimators[name] = grid_search.best_estimator_


    def simulate_journey(self, cv: Optional[Any] = None) -> Population:
        """
        Simulate the portfolio journey using WalkForward cross-validation.

        Args:
            cv (Optional[Any]): Cross-validation strategy. Defaults to WalkForward.

        Returns:
            Population: A population of dynamic portfolios.
        """
        if cv is None:
            cv = self.cv

        final_portfolios = []

        for name, best_model in self.best_estimators.items():
            dynamic_portfolio = cross_val_predict(
                estimator=best_model, 
                X=self.returns, 
                cv=cv, 
                n_jobs=-1
            )
            dynamic_portfolio.name = name
            #dynamic_portfolio.tag = '{rebalanced}'
            final_portfolios.append(dynamic_portfolio)

        return Population(final_portfolios)


    def get_journey_with_frontier(self, strategy_name: str, cv: Optional[Any] = None) -> None:
        """
        Manually runs WalkForward to capture the 'Moving Frontier' 
        and simulate the Portfolio journey with costs, plotting the results.

        Args:
            strategy_name (str): The name of the strategy to evaluate.
            cv (Optional[Any]): Cross-validation strategy. Defaults to WalkForward.
        """
        if cv is None:
            cv = self.cv        
        
        model = self.best_estimators[strategy_name]
        model.min_return = np.linspace(0.025, 0.20, 8) / 252 #frontier

        cv_portfolios = Population([])

        # Manually iterate through the "Journey"
        for i, (train_idx, test_idx) in enumerate(cv.split(self.returns)):
            X_train, X_test = self.returns.iloc[train_idx], self.returns.iloc[test_idx]
            split_label = f"Split {i+1}"
            
            # 1. Generate and label Train portfolios
            train_pop = model.fit_predict(X_train)
            train_pop.set_portfolio_params(
                name = f"{split_label} - Train",
                tag = "Train"
            )
            test_pop = model.predict(X_test)
            test_pop.set_portfolio_params(
                name = f"{split_label} - Test",
                tag = "Test"
            )
            
            cv_portfolios.extend(train_pop + test_pop)

        cv_portfolios.plot_measures(
            x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
            y=PerfMeasure.ANNUALIZED_MEAN,
            color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
            hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
        ).show()
        cv_portfolios.plot_distribution(
            measure_list=[RatioMeasure.CVAR_RATIO, RatioMeasure.ANNUALIZED_SHARPE_RATIO],
            tag_list=["Train", "Test"], n_bins=40
        ).show()


    def evaluate_robustness_combinatorial(
        self, 
        n_folds: int, 
        n_test_folds: int, 
        purged_size: int = 10, 
        embargo_size: int = 10
    ) -> Population:
        """
        Evaluate strategy robustness using Combinatorial Purged Cross-Validation (CPCV).

        Args:
            n_folds (int): Total number of folds.
            n_test_folds (int): Number of folds in each test split.
            purged_size (int): Number of observations to purge.
            embargo_size (int): Number of observations to embargo.

        Returns:
            Population: A population of portfolios from the CPCV splits.
        """
        cpcv = CombinatorialPurgedCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            purged_size=purged_size,
            embargo_size=embargo_size
        )

        final_portfolios = Population([])

        for name, best_model in self.best_estimators.items():
            dynamic_portfolio = cross_val_predict(
                estimator=best_model, 
                X=self.returns, 
                cv=cpcv, 
                n_jobs=-1,
                portfolio_params={"tag": name}
            )
            final_portfolios.extend(dynamic_portfolio)

        return final_portfolios


    def evaluate_robustness_randomised(
        self, 
        cv: Optional[WalkForward] = None, 
        n_subsamples: int = 100, 
        asset_subset_size: int = 10, 
        window_size: Optional[int] = None
    ) -> Population:
        """
        Evaluate strategy robustness using Multiple Randomised Cross-Validation.

        Args:
            cv (Optional[WalkForward]): WalkForward cross-validation instance. Defaults to self.cv.
            n_subsamples (int): Number of subsamples.
            asset_subset_size (int): Size of the asset subset.
            window_size (Optional[int]): Window size for sampling.

        Returns:
            Population: A population of portfolios from the randomised splits.
        """
        if cv is None:
            cv = self.cv
        else:
            if not isinstance(cv, WalkForward):
                raise TypeError("cv must be an instance of WalkForward")

        mrcv = MultipleRandomizedCV(
            walk_forward=cv,
            n_subsamples=n_subsamples,
            asset_subset_size=asset_subset_size,
            window_size=window_size,
            random_state=0
        )

        final_portfolios = Population([])

        for name, best_model in self.best_estimators.items():
            dynamic_portfolio = cross_val_predict(
                estimator=best_model, 
                X=self.returns, 
                cv=mrcv, 
                n_jobs=-1,
                portfolio_params={"tag": name}
            )
            final_portfolios.extend(dynamic_portfolio)

        return final_portfolios