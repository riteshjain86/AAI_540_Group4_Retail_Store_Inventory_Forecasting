"""
Standalone SageMaker training script for retail demand forecasting.

This script is uploaded to S3 and run inside the managed training container,
so it is self-contained (it does not import the project's `src` package).
It supports the two model families produced by the local pipeline:

  - 'lstm': a stacked LSTM regressor (TensorFlow/Keras) over lag/rolling/
    calendar feature sequences
  - 'ridge': a Ridge regression baseline on the same tabular features

The chosen family is passed as the `--model_type` hyperparameter so the same
script backs the training job regardless of which model the design document's
comparison selects.

When a `test` channel is provided, the trained artifact is re-loaded and
scored on the held-out test split alongside a seasonal-naive baseline, and
the result is written as `evaluation.json` inside the model artifact. The
SageMaker Pipeline's evaluation step and quality gate read that file, so the
release decision never needs to unpickle the model itself.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = 'Units Sold'
GROUP_COLS = ['Store ID', 'Product ID']
CATEGORICAL_COLS = ['Category', 'Region', 'Weather Condition', 'Seasonality'] + GROUP_COLS
LAG_DAYS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Static (data-independent) numeric feature column order, shared by
# engineer_features() and the header-less batch transform CSV contract
# (batch_transform.py's input_fn must assign these same names in this same
# order to the columns it reads, since the uploaded CSV has no header row).
NUMERIC_COLS = ['Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion',
                'day_of_week', 'month', 'quarter', 'year', 'week_of_year', 'is_weekend']
NUMERIC_COLS += [f'{TARGET_COLUMN}_lag_{lag}' for lag in LAG_DAYS]
NUMERIC_COLS += [f'{TARGET_COLUMN}_rolling_mean_{w}' for w in ROLLING_WINDOWS]
NUMERIC_COLS += [f'{TARGET_COLUMN}_rolling_std_{w}' for w in ROLLING_WINDOWS]

# Seasonal-naive baseline: the value observed 7 days earlier (weekly seasonality)
SEASONAL_NAIVE_COL = f'{TARGET_COLUMN}_lag_7'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'ridge'])
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--lstm_units_1', type=int, default=64)
    parser.add_argument('--lstm_units_2', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ridge_alpha', type=float, default=1.0)
    parser.add_argument('--random_state', type=int, default=42)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    return parser.parse_args()


def read_channel(channel: str) -> pd.DataFrame:
    files = sorted(Path(channel).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No Parquet input files in {channel}')
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror src/feature_engineering/feature_engineering.py without importing it."""
    df = df.sort_values(GROUP_COLS + ['Date']).reset_index(drop=True)
    dt = pd.to_datetime(df['Date']).dt
    df['day_of_week'] = dt.dayofweek
    df['month'] = dt.month
    df['quarter'] = dt.quarter
    df['year'] = dt.year
    df['week_of_year'] = dt.isocalendar().week.astype(int)
    df['is_weekend'] = (dt.dayofweek >= 5).astype(int)

    grouped_target = df.groupby(GROUP_COLS)[TARGET_COLUMN]
    for lag in LAG_DAYS:
        df[f'{TARGET_COLUMN}_lag_{lag}'] = grouped_target.shift(lag)
    shifted = grouped_target.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = shifted.groupby([df[c] for c in GROUP_COLS]).rolling(window=window, min_periods=window)
        df[f'{TARGET_COLUMN}_rolling_mean_{window}'] = rolling.mean().reset_index(level=list(range(len(GROUP_COLS))), drop=True)
        df[f'{TARGET_COLUMN}_rolling_std_{window}'] = rolling.std().reset_index(level=list(range(len(GROUP_COLS))), drop=True)

    feature_cols = list(NUMERIC_COLS)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def build_sequences(df, feature_cols, sequence_length, return_index=False):
    X_list, y_list, row_index = [], [], []
    feature_values = df[feature_cols].to_numpy(dtype=np.float32)
    target_values = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    for _, idx in df.groupby(GROUP_COLS, sort=False).indices.items():
        idx = np.sort(idx)
        if len(idx) <= sequence_length:
            continue
        gf, gt = feature_values[idx], target_values[idx]
        for start in range(len(idx) - sequence_length):
            end = start + sequence_length
            X_list.append(gf[start:end])
            y_list.append(gt[end])
            row_index.append(idx[end])
    X, y = np.stack(X_list), np.array(y_list, dtype=np.float32)
    if return_index:
        return X, y, np.array(row_index)
    return X, y


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.clip(np.asarray(y_pred, dtype=float), 0, None)
    denom = np.sum(np.abs(y_true))
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'wape': float(np.sum(np.abs(y_true - y_pred)) / denom) if denom else float('nan'),
    }


def train_ridge(train_df, val_df, feature_cols, args):
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
    ])
    model = Pipeline([('preprocess', preprocess), ('regressor', Ridge(alpha=args.ridge_alpha, random_state=args.random_state))])
    model.fit(train_df[numeric_cols + CATEGORICAL_COLS], train_df[TARGET_COLUMN])
    val_pred = model.predict(val_df[numeric_cols + CATEGORICAL_COLS])
    metrics = evaluate(val_df[TARGET_COLUMN], val_pred)
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    return metrics


def train_lstm(train_df, val_df, feature_cols, args):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks

    X_train, y_train = build_sequences(train_df, feature_cols, args.sequence_length)
    X_val, y_val = build_sequences(val_df, feature_cols, args.sequence_length)

    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_s = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape).astype(np.float32)
    X_val_s = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape).astype(np.float32)

    inputs = layers.Input(shape=(args.sequence_length, X_train.shape[-1]))
    x = layers.LSTM(args.lstm_units_1, activation='relu', return_sequences=True)(inputs)
    x = layers.Dropout(args.dropout)(x)
    x = layers.LSTM(args.lstm_units_2, activation='relu')(x)
    x = layers.Dropout(args.dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate), loss='mse', metrics=['mae'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=args.epochs,
             batch_size=args.batch_size, callbacks=[early_stopping], verbose=2)

    val_pred = model.predict(X_val_s, verbose=0).reshape(-1)
    metrics = evaluate(y_val, val_pred)

    model.save(os.path.join(args.model_dir, 'model.keras'))
    joblib.dump(scaler, os.path.join(args.model_dir, 'sequence_scaler.joblib'))
    return metrics


def predict_test(model_type, model_dir, test_df, feature_cols, sequence_length):
    """
    Score the held-out test split with the artifact saved in model_dir.

    Returns (y_true, y_pred, y_naive) aligned row-for-row, where y_naive is
    the seasonal-naive forecast for the same target rows.
    """
    if model_type == 'lstm':
        import tensorflow as tf

        model = tf.keras.models.load_model(os.path.join(model_dir, 'model.keras'))
        scaler = joblib.load(os.path.join(model_dir, 'sequence_scaler.joblib'))
        X, y_true, rows = build_sequences(test_df, feature_cols, sequence_length, return_index=True)
        X_s = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
        y_pred = model.predict(X_s, verbose=0).reshape(-1)
        y_naive = test_df[SEASONAL_NAIVE_COL].to_numpy()[rows]
        return y_true, y_pred, y_naive

    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    y_pred = model.predict(test_df[numeric_cols + CATEGORICAL_COLS])
    return test_df[TARGET_COLUMN].to_numpy(), y_pred, test_df[SEASONAL_NAIVE_COL].to_numpy()


def build_evaluation_report(model_type, y_true, y_pred, y_naive, validation_metrics) -> dict:
    """
    Assemble evaluation.json: candidate vs seasonal-naive metrics on the test
    split, plus a `regression_metrics` block in the format SageMaker Model
    Registry displays on the model package's metrics tab.
    """
    test_metrics = evaluate(y_true, y_pred)
    baseline_metrics = evaluate(y_true, y_naive)
    improvement = 1.0 - test_metrics['wape'] / baseline_metrics['wape'] if baseline_metrics['wape'] else 0.0
    return {
        'model_type': model_type,
        'n_test_rows': int(len(y_true)),
        'validation': validation_metrics,
        'test': test_metrics,
        'baseline_seasonal_naive': baseline_metrics,
        'wape_improvement_vs_baseline': float(improvement),
        'regression_metrics': {
            'mae': {'value': test_metrics['mae']},
            'rmse': {'value': test_metrics['rmse']},
            'wape': {'value': test_metrics['wape']},
            'baseline_wape': {'value': baseline_metrics['wape']},
        },
    }


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    train_df, feature_cols = engineer_features(read_channel(args.train))
    val_df, _ = engineer_features(read_channel(args.validation))

    if args.model_type == 'lstm':
        metrics = train_lstm(train_df, val_df, feature_cols, args)
    else:
        metrics = train_ridge(train_df, val_df, feature_cols, args)

    print(f"Validation metrics: {metrics}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({'model_type': args.model_type, **metrics}, f, indent=2)

    if args.test:
        test_df, _ = engineer_features(read_channel(args.test))
        y_true, y_pred, y_naive = predict_test(args.model_type, args.model_dir, test_df,
                                               feature_cols, args.sequence_length)
        report = build_evaluation_report(args.model_type, y_true, y_pred, y_naive, metrics)
        print(f"Test metrics: {report['test']} | seasonal-naive: {report['baseline_seasonal_naive']}")
        for directory in (args.model_dir, args.output_dir):
            with open(os.path.join(directory, 'evaluation.json'), 'w') as f:
                json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
