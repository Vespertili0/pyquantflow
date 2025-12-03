import pandas as pd
from backtesting import Backtest


class BacktestRunner:
    def __init__(self):
        pass

    def _validate_data(self, df):
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        return df[required_cols]

    def run(self, strategy_class, data, symbol=None, cash=10000, commission=.002, 
    trade_on_close=False, finalize_trades=True, **strategy_params):
        """
        Run backtest(s).
        
        Args:
            strategy_class: Strategy class to run.
            data: pd.DataFrame or dict. 
                  If DataFrame, must provide 'symbol'.
                  If dict, keys are symbols, values are DataFrames.
            symbol: Symbol name if data is a DataFrame.
            cash: Initial cash.
            commission: Commission rate.
            trade_on_close: Whether to trade on close.
            finalize_trades: Whether to finalize trades.
            **strategy_params: Parameters for the strategy.
        """
        results = {}
        
        # Normalize input to a dictionary of {symbol: dataframe}
        data_map = {}
        if isinstance(data, pd.DataFrame):
            if not symbol:
                raise ValueError("Symbol must be provided when passing a single DataFrame.")
            data_map[symbol] = data
        elif isinstance(data, dict):
            data_map = data
        else:
            raise ValueError("Data must be a pandas DataFrame or a dictionary of DataFrames.")

        for sym, df in data_map.items():
            print(f"Running backtest for {sym}...")
            try:
                df = self._validate_data(df)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                
            #    if df.index.tz is not None:
            #        df.index = df.index.tz_localize(None)
                
                bt = Backtest(df, strategy_class, cash=cash, commission=commission
                trade_on_close=trade_on_close, finalize_trades=finalize_trades)
                stats = bt.run(**strategy_params)
                
                # Convert stats to dictionary for easier handling
                # stats is a pd.Series, to_dict() works well
                results[sym] = stats.to_dict()
                
                print(f"Finished {sym}: Return {stats['Return [%]']:.2f}%")
            except Exception as e:
                print(f"Error running backtest for {sym}: {e}")
                results[sym] = {'Error': str(e)}

        return results
