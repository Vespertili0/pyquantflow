import pandas as pd
from backtesting import Backtest
from ..db.backtest_results import BacktestResultsDB
from ..config import BACKTEST_RESULTS_DB

class BacktestRunner:
    def __init__(self, results_db_path=BACKTEST_RESULTS_DB):
        self.results_db_path = results_db_path

    def _validate_data(self, df):
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        return df[required_cols]

    def run(self, strategy_class, data, symbol=None, cash=10000, commission=.002, **strategy_params):
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
            **strategy_params: Parameters for the strategy.
        """
        aggregated_results = []
        
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

        with BacktestResultsDB(self.results_db_path) as results_db:
            for sym, df in data_map.items():
                print(f"Running backtest for {sym}...")
                try:
                    df = self._validate_data(df)
                    
                    bt = Backtest(df, strategy_class, cash=cash, commission=commission)
                    stats = bt.run(**strategy_params)
                    
                    # Extract metrics
                    start_date = stats['Start'].isoformat()
                    end_date = stats['End'].isoformat()
                    return_pct = stats['Return [%]']
                    sharpe = stats['Sharpe Ratio']
                    max_drawdown = stats['Max. Drawdown [%]']
                    trades = stats['# Trades']
                    duration = str(stats['Duration'])

                    # Save to DB
                    results_db.save_result(
                        strategy_class.__name__,
                        sym,
                        start_date,
                        end_date,
                        return_pct,
                        sharpe,
                        max_drawdown,
                        trades,
                        duration
                    )
                    
                    aggregated_results.append({
                        'symbol': sym,
                        'return': return_pct,
                        'sharpe': sharpe,
                        'trades': trades
                    })
                    print(f"Finished {sym}: Return {return_pct:.2f}%")
                except Exception as e:
                    print(f"Error running backtest for {sym}: {e}")

        return aggregated_results
