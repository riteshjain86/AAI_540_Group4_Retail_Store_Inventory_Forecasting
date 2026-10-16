"""
Tests for data ingestion, preprocessing and feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.data_ingestion import (
    validate_schema, chronological_cutoffs, split_data,
    DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN,
)
from src.data_preprocessing.data_preprocessing import clean_data, build_preprocessor, clip_non_negative
from src.feature_engineering.feature_engineering import (
    add_calendar_features, add_lag_rolling_features, drop_lag_rolling_warmup,
    lag_rolling_feature_names, build_sequences, CALENDAR_FEATURES,
)


def make_raw_df(n_days=60, stores=('S001', 'S002'), products=('P0001', 'P0002'), seed=0):
    """Small synthetic frame with the retail_store_inventory.csv schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=n_days)
    rows = []
    for date in dates:
        for store in stores:
            for product in products:
                rows.append({
                    'Date': date, 'Store ID': store, 'Product ID': product,
                    'Category': rng.choice(['Groceries', 'Toys']),
                    'Region': rng.choice(['North', 'South']),
                    'Inventory Level': int(rng.integers(50, 500)),
                    'Units Sold': int(rng.integers(0, 300)),
                    'Units Ordered': int(rng.integers(20, 200)),
                    'Demand Forecast': float(rng.uniform(0, 300)),
                    'Price': float(rng.uniform(10, 100)),
                    'Discount': int(rng.integers(0, 20)),
                    'Weather Condition': rng.choice(['Sunny', 'Rainy']),
                    'Holiday/Promotion': int(rng.integers(0, 2)),
                    'Competitor Pricing': float(rng.uniform(5, 100)),
                    'Seasonality': rng.choice(['Spring', 'Winter']),
                })
    return pd.DataFrame(rows)


def test_validate_schema_passes_on_valid_data():
    df = make_raw_df()
    report = validate_schema(df)
    assert report['n_rows'] == len(df)
    assert report['n_stores'] == 2
    assert report['n_products'] == 2
    assert report['n_duplicate_keys'] == 0


def test_validate_schema_rejects_missing_column():
    with pytest.raises(ValueError, match='Missing required columns'):
        validate_schema(make_raw_df().drop(columns=['Price']))


def test_validate_schema_rejects_negative_target():
    df = make_raw_df()
    df.loc[0, 'Units Sold'] = -5
    with pytest.raises(ValueError, match='negative'):
        validate_schema(df)


def test_chronological_split_is_70_15_15_and_disjoint():
    df = make_raw_df(n_days=100)
    train_df, val_df, test_df = split_data(df, val_size=0.15, test_size=0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df[DATE_COLUMN].max() < val_df[DATE_COLUMN].min()
    assert val_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min()
    # Roughly 70/15/15 of the 100 distinct dates
    assert 65 <= train_df[DATE_COLUMN].nunique() <= 75


def test_chronological_cutoffs_match_split_data():
    df = make_raw_df(n_days=50)
    train_cutoff, val_cutoff = chronological_cutoffs(df[DATE_COLUMN], 0.15, 0.15)
    train_df, val_df, test_df = split_data(df, 0.15, 0.15)
    assert train_df[DATE_COLUMN].max() == train_cutoff
    assert val_df[DATE_COLUMN].max() == val_cutoff


def test_clean_data_drops_duplicates_and_sorts():
    df = make_raw_df(n_days=5)
    duped = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(duped)
    assert len(cleaned) == len(df)
    assert cleaned.groupby(['Store ID', 'Product ID'])['Date'].apply(lambda s: s.is_monotonic_increasing).all()


def test_clip_non_negative():
    assert list(clip_non_negative(np.array([-5.0, 0.0, 3.2]))) == [0.0, 0.0, 3.2]


def test_add_calendar_features():
    df = make_raw_df(n_days=10)
    result = add_calendar_features(df, DATE_COLUMN)
    assert set(CALENDAR_FEATURES) <= set(result.columns)
    assert (result['is_weekend'] == (result[DATE_COLUMN].dt.dayofweek >= 5)).all()


def test_lag_rolling_features_only_use_past_values():
    df = clean_data(make_raw_df(n_days=30, stores=('S001',), products=('P0001',)))
    out = add_lag_rolling_features(df, [STORE_COLUMN, PRODUCT_COLUMN], TARGET_COLUMN, [1, 7], [7])

    lag1_col = f'{TARGET_COLUMN}_lag_1'
    # Row i's lag-1 feature must equal row i-1's actual target (same series)
    shifted_actual = out[TARGET_COLUMN].shift(1)
    pd.testing.assert_series_equal(out[lag1_col], shifted_actual, check_names=False)


def test_drop_lag_rolling_warmup_removes_only_nan_rows():
    df = clean_data(make_raw_df(n_days=30, stores=('S001',), products=('P0001',)))
    out = add_lag_rolling_features(df, [STORE_COLUMN, PRODUCT_COLUMN], TARGET_COLUMN, [1, 7], [7])
    cols = lag_rolling_feature_names(TARGET_COLUMN, [1, 7], [7])
    clean = drop_lag_rolling_warmup(out, cols)
    assert clean[cols].isna().sum().sum() == 0
    assert len(clean) < len(out)


def test_build_sequences_shapes_and_no_cross_series_windows():
    df = clean_data(make_raw_df(n_days=40, stores=('S001', 'S002'), products=('P0001',)))
    df = add_calendar_features(df, DATE_COLUMN)
    df = add_lag_rolling_features(df, [STORE_COLUMN, PRODUCT_COLUMN], TARGET_COLUMN, [1, 7], [7])
    cols = lag_rolling_feature_names(TARGET_COLUMN, [1, 7], [7])
    df = drop_lag_rolling_warmup(df, cols)

    feature_cols = ['Price', 'Discount'] + CALENDAR_FEATURES + cols
    seq_len = 5
    X, y, meta = build_sequences(df, feature_cols, TARGET_COLUMN, [STORE_COLUMN, PRODUCT_COLUMN], DATE_COLUMN, seq_len)

    assert X.shape[1:] == (seq_len, len(feature_cols))
    assert X.shape[0] == len(y) == len(meta)
    # Every window's store/product must be constant within meta's identity for that sample
    assert set(meta[STORE_COLUMN].unique()) <= {'S001', 'S002'}


def test_build_preprocessor_one_hot_encodes_and_ignores_unseen_categories():
    df = make_raw_df()
    pre = build_preprocessor(['Price', 'Discount'], ['Category', 'Region'])
    pre.fit(df)

    unseen = df.head(2).copy()
    unseen['Category'] = 'Furniture_Unseen'
    out = pre.transform(unseen)
    cat_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith('Category_')]
    assert out[:, cat_cols].sum() == 0
    assert len(cat_cols) == df['Category'].nunique()
