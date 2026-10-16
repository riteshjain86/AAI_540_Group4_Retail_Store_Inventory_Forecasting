"""
Tests for model training and evaluation modules.
"""

import numpy as np
import pytest

from src.data_ingestion.data_ingestion import STORE_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN, DATE_COLUMN
from src.data_preprocessing.data_preprocessing import clean_data
from src.feature_engineering.feature_engineering import (
    add_calendar_features, add_lag_rolling_features, drop_lag_rolling_warmup,
    lag_rolling_feature_names, build_sequences, CALENDAR_FEATURES,
)
from src.models.baseline import SeasonalNaiveModel
from src.models.train_model import MODEL_TYPES, build_model, fit_and_score
from src.models.predict_model import evaluate_forecast, wape, smape, evaluate_by_segment
from src.models.sequence_models import build_lstm_model, build_transformer_model, fit_sequence_scaler, apply_sequence_scaler
from tests.test_data_preprocessing import make_raw_df

NUMERIC_COLS = ['Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion'] + CALENDAR_FEATURES
CATEGORICAL_COLS = ['Category', 'Region', 'Weather Condition', 'Seasonality', STORE_COLUMN, PRODUCT_COLUMN]


def engineered_df(n_days=60):
    df = clean_data(make_raw_df(n_days=n_days))
    df = add_calendar_features(df, DATE_COLUMN)
    df = add_lag_rolling_features(df, [STORE_COLUMN, PRODUCT_COLUMN], TARGET_COLUMN, [1, 7], [7])
    cols = lag_rolling_feature_names(TARGET_COLUMN, [1, 7], [7])
    df = drop_lag_rolling_warmup(df, cols)
    return df, NUMERIC_COLS + cols


def test_seasonal_naive_predicts_lag_7():
    df, feature_cols = engineered_df()
    model = SeasonalNaiveModel(TARGET_COLUMN, lag=7)
    pred = model.predict(df)
    assert (pred == df[f'{TARGET_COLUMN}_lag_7']).all()


def test_seasonal_naive_requires_lag_column():
    model = SeasonalNaiveModel(TARGET_COLUMN, lag=7)
    with pytest.raises(ValueError):
        model.predict(engineered_df()[0].drop(columns=[f'{TARGET_COLUMN}_lag_7']))


@pytest.mark.parametrize('model_type', MODEL_TYPES)
def test_build_model_fits_and_predicts(model_type):
    df, feature_cols = engineered_df()
    X, y = df[feature_cols + CATEGORICAL_COLS], df[TARGET_COLUMN]
    model = build_model(model_type, feature_cols, CATEGORICAL_COLS)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(df),)


def test_fit_and_score_returns_val_predictions():
    df, feature_cols = engineered_df()
    split = len(df) * 3 // 4
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    result = fit_and_score('ridge', train_df[feature_cols + CATEGORICAL_COLS], train_df[TARGET_COLUMN],
                           val_df[feature_cols + CATEGORICAL_COLS], val_df[TARGET_COLUMN],
                           feature_cols, CATEGORICAL_COLS)
    assert result['val_predictions'].shape == (len(val_df),)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_model('xgboost', NUMERIC_COLS, CATEGORICAL_COLS)


def test_wape_and_smape_basic_values():
    y_true = np.array([10, 0, 20])
    y_pred = np.array([10, 0, 10])
    assert wape(y_true, y_pred) == pytest.approx(10 / 30)
    assert smape(y_true, y_pred) >= 0


def test_evaluate_forecast_keys():
    metrics = evaluate_forecast(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert set(metrics) == {'mae', 'rmse', 'wape', 'smape'}
    assert metrics['mae'] == pytest.approx(0.0)


def test_evaluate_by_segment_breaks_down_metrics():
    y_true = np.array([10, 10, 20, 20])
    y_pred = np.array([10, 10, 0, 0])
    segment = np.array(['a', 'a', 'b', 'b'])
    result = evaluate_by_segment(y_true, y_pred, segment)
    assert set(result['segment']) == {'a', 'b'}
    a_row = result[result['segment'] == 'a'].iloc[0]
    assert a_row['mae'] == pytest.approx(0.0)


def test_build_sequences_feed_lstm_and_transformer_and_scaler_roundtrip():
    df, feature_cols = engineered_df(n_days=40)
    seq_len = 5
    X, y, meta = build_sequences(df, feature_cols, TARGET_COLUMN, [STORE_COLUMN, PRODUCT_COLUMN], DATE_COLUMN, seq_len)
    assert X.shape[0] > 0

    scaler = fit_sequence_scaler(X)
    X_scaled = apply_sequence_scaler(scaler, X)
    assert X_scaled.shape == X.shape
    assert abs(X_scaled.reshape(-1, X.shape[-1]).mean(axis=0)).max() < 1e-1

    lstm = build_lstm_model(seq_len, X.shape[-1], units=(4, 2))
    lstm_pred = lstm.predict(X_scaled[:2], verbose=0)
    assert lstm_pred.shape == (2, 1)

    transformer = build_transformer_model(seq_len, X.shape[-1], num_heads=2, key_dim=4, ff_dim=8, num_blocks=1)
    transformer_pred = transformer.predict(X_scaled[:2], verbose=0)
    assert transformer_pred.shape == (2, 1)
