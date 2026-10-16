"""
Data preprocessing module for cleaning and preparing the retail inventory data.

All transformations are expressed as scikit-learn transformers so they are
fitted on the training split only and persisted together with the model.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning shared by every downstream model.

    Drops exact duplicate (Date, Store ID, Product ID) rows (keeping the
    first occurrence) and sorts rows chronologically within each store/product
    series, which lag and rolling features rely on.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned, sorted DataFrame
    """
    df = df.drop_duplicates(subset=['Date', 'Store ID', 'Product ID'], keep='first')
    return df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Build the column transformer shared by the classical (tabular) models.

    Nominal categoricals (store/product IDs, category, region, weather,
    seasonality) are one-hot encoded rather than label encoded, since they
    have no natural ordering. Categories unseen during training are ignored
    at inference time instead of raising an error.

    Args:
        numeric_cols: List of numerical column names
        categorical_cols: List of categorical column names

    Returns:
        Unfitted ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ],
        verbose_feature_names_out=False,
    )


def clip_non_negative(predictions: np.ndarray) -> np.ndarray:
    """
    Clip forecasts to zero.

    Units sold cannot be negative, but regressors (especially linear ones)
    can produce small negative predictions.

    Args:
        predictions: Raw model predictions

    Returns:
        Predictions clipped at zero
    """
    return np.clip(predictions, a_min=0, a_max=None)
