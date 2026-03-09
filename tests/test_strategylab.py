import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from skfolio.optimization import EqualWeighted, MeanRisk
from skfolio import Population
from skfolio.model_selection import WalkForward

from pyquantflow.portfolio.strategylab import StrategyLab

import os
from pyquantflow.data.database import DatabaseManager

class TestStrategyLab(unittest.TestCase):
    def setUp(self):
        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        self.returns = None

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                returns_dict = {}
                for ticker in ['FMG.AX', 'CBA.AX', 'AAPL']:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 200:
                        returns_dict[ticker] = df['Close'].pct_change().dropna()
                db_manager.conn.close()

                if len(returns_dict) >= 2:
                    # Inner join on index to get aligned returns
                    self.returns = pd.DataFrame(returns_dict).dropna()
            except Exception:
                pass

        if self.returns is None or len(self.returns) < 100:
            print("Fallback: Using synthetic returns since not enough real data was found.")
            np.random.seed(42)
            dates = pd.date_range("2023-01-01", periods=200, freq="B")
            self.returns = pd.DataFrame(
                np.random.normal(0.005, 0.01, size=(200, 3)),
                index=dates,
                columns=["Asset1", "Asset2", "Asset3"]
            )

        # Create dummy strategy dict
        self.strategy_dict = {
            "EqualWeighted": {
                "estimator": EqualWeighted(),
                "grid": {} # no params to search
            },
            "MeanRisk": {
                "estimator": MeanRisk(),
                "grid": {"risk_aversion": [1.0, 2.0]}
            }
        }

        self.lab = StrategyLab(returns=self.returns, strategy_dict=self.strategy_dict)
        # Use a smaller train_size/test_size for testing
        self.lab.cv = WalkForward(train_size=50, test_size=10)

    def test_init(self):
        """Test initialization of StrategyLab."""
        self.assertTrue(self.lab.returns.equals(self.returns))
        self.assertEqual(self.lab.strategy_dict, self.strategy_dict)
        self.assertEqual(self.lab.best_estimators, {})
        self.assertIsInstance(self.lab.cv, WalkForward)

    def test_search_strategy_hyperparameters(self):
        """Test search_strategy_hyperparameters sets best_estimators."""
        # We can use a simple scoring metric or None. In skfolio, ratio measure is typically used.
        # But we can just use the default scoring.
        # We'll patch GridSearchCV to speed it up or avoid complex fits if we wanted,
        # but skfolio estimators fit quickly on small data.
        def mock_scoring(estimator, X):
            # dummy scorer
            return 1.0

        self.lab.search_strategy_hyperparameters(scoring=mock_scoring)

        # Check if best estimators are set
        self.assertIn("EqualWeighted", self.lab.best_estimators)
        self.assertIn("MeanRisk", self.lab.best_estimators)

        self.assertIsInstance(self.lab.best_estimators["EqualWeighted"], EqualWeighted)
        self.assertIsInstance(self.lab.best_estimators["MeanRisk"], MeanRisk)

    def test_simulate_journey(self):
        """Test simulate_journey returns a Population of portfolios."""
        # First we need to populate best_estimators
        self.lab.best_estimators = {
            "EqualWeighted": EqualWeighted(),
            "MeanRisk": MeanRisk(risk_aversion=1.0)
        }

        population = self.lab.simulate_journey()

        self.assertIsInstance(population, Population)
        self.assertEqual(len(population), 2) # 2 strategies
        self.assertEqual(population[0].name, "EqualWeighted")
        self.assertEqual(population[1].name, "MeanRisk")

    @patch("skfolio.population.Population.plot_distribution")
    @patch("skfolio.population.Population.plot_measures")
    def test_get_journey_with_frontier(self, mock_plot_measures, mock_plot_dist):
        """Test get_journey_with_frontier runs and plots are mocked out."""
        # Setup mock best_estimators
        self.lab.best_estimators = {
            "MeanRisk": MeanRisk()
        }

        # We need to mock .show() on the return of plot_measures and plot_distribution
        mock_fig1 = MagicMock()
        mock_plot_measures.return_value = mock_fig1

        mock_fig2 = MagicMock()
        mock_plot_dist.return_value = mock_fig2

        # Mock out the MeanRisk fit to avoid solver errors with real noisy data
        with patch('skfolio.optimization.MeanRisk.fit', autospec=True) as mock_fit:
            mock_fit.return_value = self.lab.best_estimators["MeanRisk"]
            # We also mock predict so it returns a valid portfolio object
            with patch('skfolio.optimization.MeanRisk.predict', autospec=True) as mock_predict:
                from skfolio import Portfolio, Population
                mock_predict.return_value = Population([Portfolio(
                    X=np.zeros((10, len(self.returns.columns))),
                    weights=np.array([1.0/len(self.returns.columns)] * len(self.returns.columns))
                )])
                self.lab.get_journey_with_frontier("MeanRisk")

        # Verify that plot methods were called
        mock_plot_measures.assert_called_once()
        mock_plot_dist.assert_called_once()

        # Verify that show() was called on the figures
        mock_fig1.show.assert_called_once()
        mock_fig2.show.assert_called_once()

    def test_evaluate_robustness_combinatorial(self):
        """Test evaluate_robustness_combinatorial using CPCV."""
        self.lab.best_estimators = {
            "EqualWeighted": EqualWeighted()
        }

        # Need enough n_folds for 200 samples
        population = self.lab.evaluate_robustness_combinatorial(
            n_folds=3,
            n_test_folds=2,
            purged_size=2,
            embargo_size=2
        )

        self.assertIsInstance(population, Population)
        # 3 folds choose 1 = 3 paths
        self.assertTrue(len(population) > 0)

    def test_evaluate_robustness_randomised(self):
        """Test evaluate_robustness_randomised using MRCV."""
        self.lab.best_estimators = {
            "EqualWeighted": EqualWeighted()
        }

        # MRCV subsample calculation is combination of num assets choose asset_subset_size
        # For 3 synthetic assets, C(3,2) = 3. For 2 real assets, C(2,2) = 1.
        # But skfolio requires n_subsamples >= 2.
        # So we need at least C(n, k) >= 2, which means we might need k=1 if n=2
        # C(2, 1) = 2.
        n_assets = len(self.returns.columns)
        import math

        # Default to 2
        asset_subset_size = min(2, n_assets)
        max_subsamples = math.comb(n_assets, asset_subset_size)

        if max_subsamples < 2:
            # Drop subset size to 1 to get more combinations
            asset_subset_size = max(1, n_assets - 1)
            max_subsamples = math.comb(n_assets, asset_subset_size)

        n_subsamples = max(2, min(2, max_subsamples))

        population = self.lab.evaluate_robustness_randomised(
            n_subsamples=n_subsamples, # small subsamples for test
            asset_subset_size=asset_subset_size,
            window_size=100
        )

        self.assertIsInstance(population, Population)
        self.assertTrue(len(population) > 0)

if __name__ == '__main__':
    unittest.main()
