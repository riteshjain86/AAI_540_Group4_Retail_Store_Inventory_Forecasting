"""
Deep-learning sequence models (LSTM and Transformer) for demand forecasting.

Both models consume the same input: a (sequence_length, n_features) window of
per-day features for a single (store, product) series, and predict the next
day's target. They are trained as "global" models across every series at
once, which is the standard approach for many short, related time series
rather than fitting a separate model per series.
"""

from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers, callbacks


def fit_sequence_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on the training sequences only.

    Args:
        X_train: Training sequences, shape (n_samples, sequence_length, n_features)

    Returns:
        Fitted StandardScaler (operates on the flattened last dimension)
    """
    scaler = StandardScaler()
    n_samples, seq_len, n_features = X_train.shape
    scaler.fit(X_train.reshape(-1, n_features))
    return scaler


def apply_sequence_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply a fitted scaler to a batch of sequences, preserving their shape."""
    n_samples, seq_len, n_features = X.shape
    return scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features).astype(np.float32)


def build_lstm_model(sequence_length: int, n_features: int, units=(64, 32),
                     dropout: float = 0.2, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Build a stacked LSTM regressor.

    Architecture: LSTM(units[0], return_sequences=True) -> Dropout ->
    LSTM(units[1]) -> Dropout -> Dense(32, relu) -> Dense(1).

    Args:
        sequence_length: Number of timesteps per input window
        n_features: Number of features per timestep
        units: Hidden units for each of the two LSTM layers
        dropout: Dropout rate applied after each LSTM layer
        learning_rate: Adam learning rate

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(sequence_length, n_features))
    x = layers.LSTM(units[0], activation='relu', return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units[1], activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs, name='lstm_forecaster')
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


def _positional_encoding(sequence_length: int, d_model: int) -> tf.Tensor:
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""
    positions = np.arange(sequence_length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles = positions * angle_rates
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(angles[np.newaxis, ...], dtype=tf.float32)


def _transformer_block(x, num_heads: int, key_dim: int, ff_dim: int, dropout: float):
    """One pre-norm Transformer encoder block: self-attention + feed-forward, each with a residual connection."""
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff_out = layers.Dense(ff_dim, activation='relu')(x)
    ff_out = layers.Dense(x.shape[-1])(ff_out)
    ff_out = layers.Dropout(dropout)(ff_out)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff_out)


def build_transformer_model(sequence_length: int, n_features: int, num_heads: int = 4,
                            key_dim: int = 32, ff_dim: int = 64, num_blocks: int = 2,
                            dropout: float = 0.1, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Build a small Transformer-encoder regressor.

    Architecture: input projection -> positional encoding -> `num_blocks`
    Transformer encoder blocks (multi-head self-attention + feed-forward) ->
    global average pooling -> Dense(1).

    Args:
        sequence_length: Number of timesteps per input window
        n_features: Number of features per timestep
        num_heads: Number of attention heads
        key_dim: Dimensionality of each attention head
        ff_dim: Hidden size of the feed-forward sublayer
        num_blocks: Number of stacked encoder blocks
        dropout: Dropout rate used in attention and feed-forward sublayers
        learning_rate: Adam learning rate

    Returns:
        Compiled Keras model
    """
    d_model = num_heads * key_dim
    inputs = layers.Input(shape=(sequence_length, n_features))
    x = layers.Dense(d_model)(inputs)
    x = x + _positional_encoding(sequence_length, d_model)

    for _ in range(num_blocks):
        x = _transformer_block(x, num_heads, key_dim, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs, name='transformer_forecaster')
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


SEQUENCE_MODEL_BUILDERS = {
    'lstm': build_lstm_model,
    'transformer': build_transformer_model,
}


def train_sequence_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, dl_config: Dict[str, Any],
                         random_state: int = 42) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Scale, build and train an LSTM or Transformer sequence model with early stopping.

    Args:
        model_type: 'lstm' or 'transformer'
        X_train, y_train: Training sequences/targets
        X_val, y_val: Validation sequences/targets
        dl_config: The `training.deep_learning` section of config.yaml
        random_state: Random seed

    Returns:
        Tuple of (trained Keras model, dict with the fitted scaler and history)
    """
    tf.random.set_seed(random_state)
    scaler = fit_sequence_scaler(X_train)
    X_train_scaled = apply_sequence_scaler(scaler, X_train)
    X_val_scaled = apply_sequence_scaler(scaler, X_val)

    builder = SEQUENCE_MODEL_BUILDERS[model_type]
    model_kwargs = dl_config.get(model_type, {})
    model = builder(X_train.shape[1], X_train.shape[2],
                    learning_rate=dl_config['learning_rate'], **model_kwargs)

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=dl_config['early_stopping_patience'], restore_best_weights=True,
    )
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=dl_config['epochs'], batch_size=dl_config['batch_size'],
        callbacks=[early_stopping], verbose=0,
    )
    return model, {'scaler': scaler, 'history': history.history}


def predict_sequence_model(model: tf.keras.Model, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Scale inputs with the training scaler and return 1-D predictions."""
    X_scaled = apply_sequence_scaler(scaler, X)
    return model.predict(X_scaled, verbose=0).reshape(-1)
