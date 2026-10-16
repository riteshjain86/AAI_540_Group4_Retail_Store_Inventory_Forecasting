"""
Data ingestion module for loading, validating and splitting the raw
retail store inventory data.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Union

# Expected schema of the Kaggle Retail Store Inventory Forecasting dataset
# (retail_store_inventory.csv: 5 stores x 20 products x 731 days).
DATE_COLUMN = 'Date'
STORE_COLUMN = 'Store ID'
PRODUCT_COLUMN = 'Product ID'
TARGET_COLUMN = 'Units Sold'

NUMERIC_COLUMNS = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast',
                   'Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion']
CATEGORICAL_COLUMNS = ['Category', 'Region', 'Weather Condition', 'Seasonality']
ID_COLUMNS = [STORE_COLUMN, PRODUCT_COLUMN]
EXPECTED_COLUMNS = [DATE_COLUMN] + ID_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the raw retail inventory data from CSV or Parquet.

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame containing the loaded data, with Date parsed as datetime
    """
    file_path = Path(file_path)

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that the DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if validation passes, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Run schema and data-quality checks on the raw retail inventory data.

    Checks column presence, numeric types, nulls, duplicate (date, store,
    product) keys and negative values in columns that should not have them.

    Args:
        df: Raw DataFrame

    Returns:
        Dictionary summarising the data-quality report
    """
    validate_data(df, EXPECTED_COLUMNS)

    non_numeric = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Expected numeric columns have non-numeric types: {non_numeric}")

    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    negative_target = int((df[TARGET_COLUMN] < 0).sum())
    if negative_target:
        raise ValueError(f"{negative_target} rows have negative '{TARGET_COLUMN}' values")

    key_duplicates = int(df.duplicated(subset=[DATE_COLUMN] + ID_COLUMNS).sum())

    return {
        'n_rows': len(df),
        'n_stores': int(df[STORE_COLUMN].nunique()),
        'n_products': int(df[PRODUCT_COLUMN].nunique()),
        'date_min': str(df[DATE_COLUMN].min().date()),
        'date_max': str(df[DATE_COLUMN].max().date()),
        'n_duplicate_keys': key_duplicates,
        'target_mean': round(float(df[TARGET_COLUMN].mean()), 2),
        'target_zero_rate': round(float((df[TARGET_COLUMN] == 0).mean()), 4),
    }


def chronological_cutoffs(dates, val_size: float = 0.15, test_size: float = 0.15
                          ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute the two date cutoffs that separate train / validation / test.

    Exposed separately from split_data so that other row-granularities of the
    same dataset (e.g. the sequences built for the LSTM/Transformer models)
    can be assigned to the same chronological splits consistently.

    Args:
        dates: Iterable of (unique) dates covering the full dataset
        val_size: Fraction of the date range used for validation
        test_size: Fraction of the date range used for the final test

    Returns:
        Tuple of (train_cutoff, val_cutoff); train is dates <= train_cutoff,
        validation is train_cutoff < dates <= val_cutoff, test is dates > val_cutoff
    """
    dates = sorted(set(dates))
    n_dates = len(dates)
    n_test = max(1, int(round(n_dates * test_size)))
    n_val = max(1, int(round(n_dates * val_size)))
    n_train = n_dates - n_val - n_test
    if n_train <= 0:
        raise ValueError("val_size + test_size leave no dates for training")
    return dates[n_train - 1], dates[n_train + n_val - 1]


def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split.

    Unlike a classification problem, rows are *not* shuffled: every store and
    product shares the same calendar, so a single global date cutoff separates
    the splits and every lag/rolling feature at inference time only looks at
    data strictly before its own split's start.

    Args:
        df: Full DataFrame, one row per (date, store, product)
        val_size: Fraction of the date range used for validation
        test_size: Fraction of the date range used for the final test

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_cutoff, val_cutoff = chronological_cutoffs(df[DATE_COLUMN], val_size, test_size)

    train_df = df[df[DATE_COLUMN] <= train_cutoff].reset_index(drop=True)
    val_df = df[(df[DATE_COLUMN] > train_cutoff) & (df[DATE_COLUMN] <= val_cutoff)].reset_index(drop=True)
    test_df = df[df[DATE_COLUMN] > val_cutoff].reset_index(drop=True)
    return train_df, val_df, test_df
