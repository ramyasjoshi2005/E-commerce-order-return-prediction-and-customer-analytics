"""Project paths and modeling constants.

All training and inference code should import from here so feature names,
split dates, and history rules stay identical.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODELS = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

# Existing labeled order table produced by the earlier pipeline.
# Global Superstore CSVs in data/raw cannot be joined to returns.csv (see README).
LABELED_ORDERS_PATH = DATA_PROCESSED / "features.csv"

ORDERS_CLEAN_PATH = DATA_PROCESSED / "orders_clean.csv"
FEATURES_PATH = DATA_PROCESSED / "model_features.csv"
SQLITE_PATH = DATA_PROCESSED / "returns.db"

MODEL_PATH = MODELS / "best_model.joblib"
PREPROCESSOR_INFO_PATH = MODELS / "feature_spec.json"

RANDOM_STATE = 42

# First-time customers have no prior order. Trees can split on this sentinel.
FIRST_ORDER_DAYS = 9999

TARGET = "returned"

ID_COLS = ["order_id", "customer_id", "order_date"]

NUMERIC_FEATURES = [
    "order_value",
    "quantity",
    "discount",
    "max_discount",
    "number_of_products",
    "number_of_categories",
    "is_weekend",
    "previous_order_count",
    "previous_return_count",
    "historical_return_rate",
    "historical_average_order_value",
    "days_since_previous_order",
    "is_first_order",
]

CATEGORICAL_FEATURES = [
    "region",
    "segment",
    "order_month",
    "order_day_of_week",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Temporal split uses the actual calendar range of this labeled table (2018-2021).
TRAIN_END = "2019-12-31"
VAL_END = "2020-12-31"
