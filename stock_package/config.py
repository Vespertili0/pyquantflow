import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, 'db_files')

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

MARKET_DATA_DB = os.path.join(DB_DIR, 'market_data.db')
MODEL_REGISTRY_DB = os.path.join(DB_DIR, 'models.db')
BACKTEST_RESULTS_DB = os.path.join(DB_DIR, 'results.db')
