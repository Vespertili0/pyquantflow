from backtesting import Strategy
import pandas as pd

class StrategyFactory:
    def __init__(self, indicators=None, ml_model=None, rules=None):
        self.indicators = indicators or []
        self.ml_model = ml_model
        self.rules = rules or []

    def create(self, name="CustomStrategy"):
        factory = self

        class CustomStrategy(Strategy):
            def init(self):
                # Initialize indicators
                for ind in factory.indicators:
                    setattr(self, ind.__name__, self.I(ind, self.data.Close))

                # Initialize ML model if provided
                if factory.ml_model:
                    self.model = factory.ml_model
                    X = pd.DataFrame({"Close": self.data.Close})
                    y = (self.data.Close.shift(-1) > self.data.Close).astype(int)
                    self.model.fit(X[:-1], y[:-1])

            def next(self):
                # If ML model is present, use it
                if factory.ml_model:
                    X_new = pd.DataFrame({"Close": [self.data.Close[-1]]})
                    pred = self.model.predict(X_new)[0]
                    if pred == 1:
                        self.buy()
                    else:
                        self.sell()
                else:
                    # Apply rule‑based logic
                    for rule in factory.rules:
                        rule(self)

        CustomStrategy.__name__ = name
        return CustomStrategy
