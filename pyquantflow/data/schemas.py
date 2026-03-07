import pandera as pa
import pandas as pd
from typing import Optional

# Core Financial DataFrame Schema for a single asset (e.g., passing into backtesting.py)
OHLCVSchema = pa.DataFrameSchema(
    {
        "Open": pa.Column(float, coerce=True),
        "High": pa.Column(float, coerce=True),
        "Low": pa.Column(float, coerce=True),
        "Close": pa.Column(float, coerce=True),
        "Volume": pa.Column(float, coerce=True),
        # Allowing other columns (like indicators or targets) that may exist
    },
    # Use pandas native string to allow arbitrary precision datetime64 with timezone
    index=pa.Index("datetime64[ns, UTC]", name="datetime", coerce=True),
    strict=False  # Allows extra columns without failing
)

# MultiAsset DataFrame Schema used predominantly by AssetOrganiser and Classifiers
MultiAssetSchema = pa.DataFrameSchema(
    {
        "Open": pa.Column(float, coerce=True),
        "High": pa.Column(float, coerce=True),
        "Low": pa.Column(float, coerce=True),
        "Close": pa.Column(float, coerce=True),
        "Volume": pa.Column(float, coerce=True),
    },
    index=pa.MultiIndex(
        [
            pa.Index("datetime64[ns, UTC]", name="datetime", coerce=True),
            pa.Index(str, name="ticker")
        ]
    ),
    strict=False  # Allows extra feature columns
)

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validates and coerces a DataFrame to the standard OHLCV schema."""
    if df.index.name is None:
        df.index.name = "datetime"
    return OHLCVSchema.validate(df)

def validate_multi_asset(df: pd.DataFrame) -> pd.DataFrame:
    """Validates and coerces a DataFrame to the standard MultiAsset schema."""
    return MultiAssetSchema.validate(df)
