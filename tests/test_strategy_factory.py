import unittest
from backtesting import Strategy, Backtest
from pyquantflow.strategies.basestrategy import StrategyFactory
import pandas as pd
import numpy as np

# Define a standalone indicator function because lambdas don't have __name__
def SMA10(values):
    return pd.Series(values).rolling(10).mean()

class TestStrategyFactory(unittest.TestCase):
    def test_create_strategy_with_rules(self):
        # Define a simple rule
        def simple_rule(strategy):
            # Access indicator created in init
            # The factory sets the attribute name to the function name
            sma = strategy.SMA10

            # Check if we have enough data
            if len(sma) < 10 or pd.isna(sma[-1]):
                return

            # Simple logic
            if strategy.data.Close[-1] > sma[-1]:
                 if not strategy.position:
                     strategy.buy()
            elif strategy.data.Close[-1] < sma[-1]:
                 if strategy.position:
                     strategy.sell()

        # Initialize factory with an indicator and a rule
        factory = StrategyFactory(indicators=[SMA10], rules=[simple_rule])
        MyStrategy = factory.create(name="MyRuleStrategy")

        self.assertTrue(issubclass(MyStrategy, Strategy))
        self.assertEqual(MyStrategy.__name__, "MyRuleStrategy")

        # Test execution
        import os
        from pyquantflow.data.database import DatabaseManager

        source_db_path = os.path.join(os.path.dirname(__file__), "stocks.db")
        data = None

        if os.path.exists(source_db_path):
            try:
                db_manager = DatabaseManager(db_path=source_db_path)
                for ticker in ['FMG.AX', 'CBA.AX']:
                    df = db_manager.get_data(ticker)
                    if not df.empty and len(df) >= 100:
                        data = df
                        break
                db_manager.conn.close()
            except Exception:
                pass

        if data is None:
            print("Fallback: Using synthetic data since no real data was found.")
            np.random.seed(42)
            returns = np.random.normal(0, 0.01, 100)
            price_path = 10 * np.exp(np.cumsum(returns))

            data = pd.DataFrame({
                'Open': price_path,
                'High': price_path * 1.01,
                'Low': price_path * 0.99,
                'Close': price_path,
                'Volume': 1000
            }, index=pd.date_range('2023-01-01', periods=100))

        bt = Backtest(data, MyStrategy, cash=10000, commission=.002)
        stats = bt.run()

        self.assertIsNotNone(stats)
        # Check that we made some trades or at least didn't crash
        # print(stats)

if __name__ == '__main__':
    unittest.main()
