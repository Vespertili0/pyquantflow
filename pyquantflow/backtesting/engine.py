from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
from backtesting import Backtest
from ..data.assetorganiser import AssetOrganiser
from ..data.schemas import validate_ohlcv


class BacktestRunner:
    def __init__(self):
        pass

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates using pandera schemas to ensure typing and index structure."""
        try:
            return validate_ohlcv(df)
        except Exception as e:
            raise ValueError(f"Data schema validation failed: {e}")

    def run(self, strategy_class: type, data: Optional[pd.DataFrame | Dict[str, pd.DataFrame]] = None,
            asset_organiser: Optional[AssetOrganiser] = None, symbols: Optional[List[str]] = None,
            cash: float = 10000, commission: float | Tuple[float, float] = (3, 0), 
            trade_on_close: bool = False, finalize_trades: bool = True, **strategy_params) -> Dict[str, Any]:
        """
        Run backtest(s).
        
        Args:
            strategy_class: Strategy class to run.
            data: pd.DataFrame or dict. 
                  If DataFrame, must provide 'symbols' as a single-element list or infer from index.
                  If dict, keys are symbols, values are DataFrames.
            asset_organiser: AssetOrganiser instance containing transformed test data.
            symbols: List of symbol names to run backtests on. Required if data is a DataFrame,
                     or if you want to subset symbols from an AssetOrganiser.
            cash: Initial cash.
            commission: Commission rate.
            trade_on_close: Whether to trade on close.
            finalize_trades: Whether to finalize trades.
            **strategy_params: Parameters for the strategy.
            
        Returns:
            Dict[str, Any]: Dictionary mapping symbols to backtest results.
        """
        results = {}
        
        # Determine data source and normalize to a dictionary of {symbol: dataframe}
        data_map: Dict[str, pd.DataFrame] = {}
        
        if asset_organiser is not None:
            if data is not None:
                raise ValueError("Cannot provide both 'data' and 'asset_organiser'. Choose one.")
            
            # Use AssetOrganiser
            multiasset_test_data = asset_organiser.get_transformed_multiasset_testdata()
            available_symbols = getattr(multiasset_test_data.index.get_level_values('ticker'), 'unique', lambda: [])()
            if callable(available_symbols):
                available_symbols = available_symbols()
            else:
                 available_symbols = list(available_symbols)

            target_symbols = symbols if symbols is not None else available_symbols
            
            for sym in target_symbols:
                if sym in available_symbols:
                    data_map[sym] = asset_organiser.get_transformed_test_ticker(sym)
                else:
                    print(f"Warning: Symbol '{sym}' not found in AssetOrganiser test data.")
                    
        elif data is not None:
            # Legacy raw data handling
            if isinstance(data, pd.DataFrame):
                symbol = symbols[0] if (symbols and len(symbols) > 0) else None
                if not symbol:
                    raise ValueError("Symbols list must be provided when passing a single DataFrame.")
                data_map[symbol] = data
            elif isinstance(data, dict):
                data_map = data
            else:
                raise ValueError("Data must be a pandas DataFrame or a dictionary of DataFrames.")
        else:
             raise ValueError("Must provide either 'data' or 'asset_organiser'.")

        for sym, df in data_map.items():
            print(f"Running backtest for {sym}...")
            try:
                df = self._validate_data(df)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                
            #    if df.index.tz is not None:
            #        df.index = df.index.tz_localize(None)
                
                bt = Backtest(df, strategy_class, cash=cash, commission=commission,
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
