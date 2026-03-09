import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from skfolio.optimization import EqualWeighted, MeanRisk
from skfolio import Population
from skfolio.model_selection import WalkForward

from pyquantflow.portfolio.strategylab import StrategyLab

class TestStrategyLab(unittest.TestCase):
    def setUp(self):
        # Create dummy returns data
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

        population = self.lab.evaluate_robustness_randomised(
            n_subsamples=2, # small subsamples for test
            asset_subset_size=2,
            window_size=100
        )

        self.assertIsInstance(population, Population)
        self.assertTrue(len(population) > 0)

if __name__ == '__main__':
    unittest.main()
