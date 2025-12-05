from backtesting import Strategy

class StrategyFactory:
    def __init__(self, init_function, next_function):
        self.init_function = init_function
        self.next_function = next_function

    def create(self, name="CustomStrategy", params=None):
        factory = self

        class CustomStrategy(Strategy):
            def init(self):
                factory.init_function(self)

            def next(self):
                factory.next_function(self)

        CustomStrategy.__name__ = name

        if params:
            for key, value in params.items():
                setattr(CustomStrategy, key, value)

        return CustomStrategy
