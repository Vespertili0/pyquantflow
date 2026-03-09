import numpy as np
import pandas as pd


def pipe_indicator(df: pd.DataFrame, indicator, input_map, output_names, **kwargs) -> pd.DataFrame:
    """
    A Pandas pipe-compatible function to calculate indicators and inject them 
    back into the DataFrame.

    Args:
        df (pd.DataFrame): The input dataframe.
        func (callable): The indicator function (e.g., ICHIMOKU or talib.RSI).
        input_map (dict or list): 
            - If dict: maps function arguments to DF column names. e.g. {'high': 'High', 'low': 'Low'}
            - If list: maps positional function arguments to DF column names. e.g. ['Close']
        output_names (str or list): Names for the resulting columns. 
            - If the function returns a tuple, provide a list of names. 
            - Use None in the list to skip specific return values.
        **kwargs: Static arguments passed to the indicator function (e.g., timeperiod=14).

    Returns:
        pd.DataFrame: The dataframe with new indicator columns.
    """
    
    # 1. Prepare Data Inputs
    if isinstance(input_map, dict):
        # Pass data as Keyword Arguments (Good for functions with named inputs like ours)
        data_inputs = {arg: df[col].values for arg, col in input_map.items()}
        # Combine with static kwargs
        full_kwargs = {**data_inputs, **kwargs}
        results = indicator(**full_kwargs)
        
    elif isinstance(input_map, list) or isinstance(input_map, tuple):
        # Pass data as Positional Arguments (Good for standard TA-Lib functions like RSI)
        pos_inputs = [df[col].values for col in input_map]
        results = indicator(*pos_inputs, **kwargs)
    else:
        raise ValueError("input_map must be a dict or list/tuple")

    # 2. Handle Output Assignment
    
    # Normalize results to be iterable if it's a single value
    if not isinstance(results, tuple):
        results = (results,)
    
    # Normalize output_names to list
    if isinstance(output_names, str):
        output_names = [output_names]

    # Assign columns
    for name, data in zip(output_names, results):
        if name is not None and data is not None:
            df[name] = data
            
    return df


def restructure_map_2_multiasset_df(df_dict, key_column_name='ticker'):
    """
    Merges a dictionary of pandas DataFrames into a single DataFrame,
    adding the dictionary key as a new column.

    Parameters:
    -----------
    df_dict : dict
        A dictionary where keys are identifiers and values are pandas DataFrames.
    key_column_name : str, optional (default='source_key')
        The name of the new column that will hold the dictionary keys.

    Returns:
    --------
    pd.DataFrame
        A single concatenated DataFrame.
    """
    # 1. Handle empty input
    if not df_dict:
        return pd.DataFrame()

    # 2. Prepare a list to collect modified dataframes
    dfs_to_concat = []

    for key, df in df_dict.items():
        # Create a copy to avoid modifying the original dataframe
        temp_df = df.copy()
        # Assign the key to the new column
        temp_df[key_column_name] = key
        dfs_to_concat.append(temp_df)

    # 3. Concatenate all dataframes
    # ignore_index=True ensures a clean new index (0, 1, 2...)
    final_df = pd.concat(dfs_to_concat).reset_index()

    return final_df.dropna().set_index(["datetime", "ticker"]).sort_index()


def align_and_ffill_multiasset(df, time_level="datetime", ticker_level="ticker"):
    """
    Aligns a multi-asset dataframe to a full timestamp × ticker grid
    and forward-fills each ticker independently.
    """
    full_index = pd.MultiIndex.from_product([
        df.index.get_level_values("datetime").unique(),
        df.index.get_level_values("ticker").unique()
        ],
        names=["datetime", "ticker"]
    )
    df = df.reindex(full_index)

    df = df.groupby(level=ticker_level).ffill()

    return df.dropna()