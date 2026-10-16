"""
SageMaker real-time endpoint inference script.

Supports both model artifact types produced by the training pipeline:
  - model.joblib (classical scikit-learn Pipeline)
  - model.keras + sequence_scaler.joblib (LSTM / Transformer)

This is an optional extension to the primary batch-inference pattern
described in the design document.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """Load whichever model artifact is present in model_dir."""
    keras_path = os.path.join(model_dir, 'model.keras')
    if os.path.exists(keras_path):
        import tensorflow as tf
        model = tf.keras.models.load_model(keras_path)
        scaler = joblib.load(os.path.join(model_dir, 'sequence_scaler.joblib'))
        return {'type': 'sequence', 'model': model, 'scaler': scaler}

    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return {'type': 'tabular', 'model': model}


def input_fn(request_body, request_content_type):
    """Parse a JSON payload into a pandas DataFrame or a 3-D numpy array."""
    if request_content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    if isinstance(payload, dict) and 'sequence' in payload:
        # Pre-built sequence for the LSTM/Transformer models: shape (timesteps, features)
        return np.array(payload['sequence'], dtype=np.float32)[np.newaxis, ...]
    # Tabular request: a list of row dicts
    return pd.DataFrame(payload)


def predict_fn(input_data, model_bundle):
    """Make predictions and clip to zero (units sold cannot be negative)."""
    if model_bundle['type'] == 'sequence':
        n_samples, seq_len, n_features = input_data.shape
        scaled = model_bundle['scaler'].transform(
            input_data.reshape(-1, n_features)
        ).reshape(n_samples, seq_len, n_features).astype(np.float32)
        prediction = model_bundle['model'].predict(scaled, verbose=0).reshape(-1)
    else:
        prediction = model_bundle['model'].predict(input_data)
    return np.clip(prediction, a_min=0, a_max=None)


def output_fn(prediction, accept):
    """Format prediction output."""
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    if accept == 'text/csv':
        return ','.join(str(x) for x in prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
