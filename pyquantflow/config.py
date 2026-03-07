import os
from dataclasses import dataclass

@dataclass
class Config:
    """Centralized configuration for pyquantflow."""
    # Database paths
    db_path: str = os.getenv("PYQUANTFLOW_DB_PATH", "stocks.db")
    results_db_path: str = os.getenv("PYQUANTFLOW_RESULTS_DB_PATH", "backtest_results.db")

    # Network / API Settings
    default_interval: str = os.getenv("PYQUANTFLOW_DEFAULT_INTERVAL", "1h")
    default_start_year: int = int(os.getenv("PYQUANTFLOW_DEFAULT_START_YEAR", 2020))

# Global configuration instance
config = Config()
