"""
Feature engineering module for the retail inventory forecasting pipeline.

Calendar and lag/rolling features are computed once on the full, chronologically
sorted dataset (grouped by store/product) *before* the train/val/test split.
This is not leakage: every lag/rolling value at a given row only looks at
*earlier* dates of the same series, so a test-split row's features may be
built from train-split history, exactly as they would be at inference time in
production (today's forecast legitimately uses yesterday's actual sales).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

CALENDAR_FEATURES = ['day_of_week', 'month', 'quarter', 'year', 'week_of_year', 'is_weekend']


def add_calendar_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Add calendar features derived from the date column.

    Args:
        df: DataFrame with a datetime date column
        date_col: Name of the date column

    Returns:
        DataFrame with CALENDAR_FEATURES columns added
    """
    df = df.copy()
    dt = df[date_col].dt
    df['day_of_week'] = dt.dayofweek
    df['month'] = dt.month
    df['quarter'] = dt.quarter
    df['year'] = dt.year
    df['week_of_year'] = dt.isocalendar().week.astype(int)
    df['is_weekend'] = (dt.dayofweek >= 5).astype(int)
    return df


def lag_rolling_feature_names(target_col: str, lag_days: List[int], rolling_windows: List[int]) -> List[str]:
    """Names of the columns created by add_lag_rolling_features, for a given config."""
    lag_names = [f'{target_col}_lag_{lag}' for lag in lag_days]
    rolling_names = [f'{target_col}_rolling_mean_{w}' for w in rolling_windows]
    rolling_names += [f'{target_col}_rolling_std_{w}' for w in rolling_windows]
    return lag_names + rolling_names


def add_lag_rolling_features(df: pd.DataFrame, group_cols: List[str], target_col: str,
                             lag_days: List[int], rolling_windows: List[int]) -> pd.DataFrame:
    """
    Add per-series lag and rolling statistics of the target.

    Rolling statistics are computed on the series shifted by one day first, so
    the rolling window for a given row never includes that row's own target.

    Args:
        df: DataFrame sorted by group_cols + date (ascending)
        group_cols: Columns identifying an individual time series (e.g. store, product)
        target_col: Column to compute lag/rolling statistics on
        lag_days: List of lag offsets in days
        rolling_windows: List of rolling window sizes in days

    Returns:
        DataFrame with lag and rolling feature columns added. The first
        max(lag_days, rolling_windows) rows of each group will contain NaNs
        and should be dropped with drop_lag_rolling_warmup.
    """
    df = df.copy()
    grouped_target = df.groupby(group_cols)[target_col]

    for lag in lag_days:
        df[f'{target_col}_lag_{lag}'] = grouped_target.shift(lag)

    shifted = grouped_target.shift(1)
    for window in rolling_windows:
        rolling = shifted.groupby([df[c] for c in group_cols]).rolling(window=window, min_periods=window)
        df[f'{target_col}_rolling_mean_{window}'] = rolling.mean().reset_index(level=list(range(len(group_cols))), drop=True)
        df[f'{target_col}_rolling_std_{window}'] = rolling.std().reset_index(level=list(range(len(group_cols))), drop=True)

    return df


def drop_lag_rolling_warmup(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Drop the warm-up rows at the start of each series where lag/rolling features are NaN."""
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def build_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                    group_cols: List[str], date_col: str, sequence_length: int
                    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sliding-window sequences for the LSTM / Transformer models.

    For every (store, product) series, each window of `sequence_length`
    consecutive days of features predicts the target on the day immediately
    after the window. Windows are built only within a single series (never
    across a store/product boundary).

    Args:
        df: Feature-engineered DataFrame, sorted by group_cols + date
        feature_cols: Numeric feature columns to include at each timestep
        target_col: Column to forecast
        group_cols: Columns identifying an individual time series
        date_col: Date column, used to record the forecast date of each sample
        sequence_length: Number of past days in each input window

    Returns:
        Tuple of (X, y, meta) where X has shape (n_samples, sequence_length,
        n_features), y has shape (n_samples,), and meta is a DataFrame with
        the group identifiers and the forecast date for every sample (for
        traceability and error analysis).
    """
    X_list, y_list, meta_rows = [], [], []
    feature_values = df[feature_cols].to_numpy(dtype=np.float32)
    target_values = df[target_col].to_numpy(dtype=np.float32)

    for _, group in df.groupby(group_cols, sort=False).indices.items():
        idx = np.sort(group)
        if len(idx) <= sequence_length:
            continue
        group_features = feature_values[idx]
        group_target = target_values[idx]
        group_dates = df[date_col].to_numpy()[idx]
        group_keys = df[group_cols].to_numpy()[idx]

        for start in range(len(idx) - sequence_length):
            end = start + sequence_length
            X_list.append(group_features[start:end])
            y_list.append(group_target[end])
            meta_rows.append(list(group_keys[end]) + [group_dates[end]])

    X = np.stack(X_list) if X_list else np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows, columns=group_cols + [date_col])
    return X, y, meta
