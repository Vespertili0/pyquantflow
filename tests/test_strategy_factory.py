import unittest
from backtesting import Strategy, Backtest
from pyquantflow.strategies.basestrategy import StrategyFactory
import pandas as pd
import numpy as np

class TestStrategyFactory(unittest.TestCase):
    def test_create_strategy(self):
        # Define simple init and next functions
        def init_logic(strategy):
            # Using a simple moving average as an indicator
            # strategy.I expects a function and its arguments
            strategy.ma = strategy.I(lambda x: pd.Series(x).rolling(10).mean(), strategy.data.Close)

        def next_logic(strategy):
            # Simple logic: buy if not in position
            if not strategy.position:
                strategy.buy()
            else:
                strategy.sell()

        factory = StrategyFactory(init_logic, next_logic)
        MyStrategy = factory.create(name="MyTestStrategy", params={'param1': 100})

        self.assertTrue(issubclass(MyStrategy, Strategy))
        self.assertEqual(MyStrategy.__name__, "MyTestStrategy")
        self.assertEqual(MyStrategy.param1, 100)

        # Test if it runs in Backtest
        # Create dummy data
        np.random.seed(42)
        data = pd.DataFrame({
            'Open': np.random.rand(100) + 10,
            'High': np.random.rand(100) + 10,
            'Low': np.random.rand(100) + 10,
            'Close': np.random.rand(100) + 10,
            'Volume': np.random.rand(100) * 1000
        }, index=pd.date_range('2023-01-01', periods=100))

        bt = Backtest(data, MyStrategy, cash=10000, commission=.002)
        stats = bt.run()

        # Check if something happened
        self.assertIsNotNone(stats)
        # We can inspect stats to see if trades occurred, but basic execution is enough
        # print(stats)

if __name__ == '__main__':
    unittest.main()
