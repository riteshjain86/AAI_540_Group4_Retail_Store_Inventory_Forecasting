"""
SageMaker batch transform script for retail demand forecasting.

Batch inference is the primary deployment pattern described in the design
document: a prepared feature dataset (CSV, one row per store/product/day) is
read from S3 and forecasts are written back to the predictions prefix.
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
    """Parse batch input. CSV rows are tabular features (used by classical models).

    The uploaded batch input CSV has no header row (see
    batch_inference.py's prepare_batch_input()), so the column names/order
    must be assigned explicitly here to match exactly - the ColumnTransformer
    inside the fitted Pipeline selects columns by name, not position.
    """
    if request_content_type == 'text/csv':
        from io import StringIO
        from train import NUMERIC_COLS, CATEGORICAL_COLS
        df = pd.read_csv(StringIO(request_body), header=None)
        df.columns = NUMERIC_COLS + CATEGORICAL_COLS
        return df
    if request_content_type == 'application/json':
        payload = json.loads(request_body)
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_bundle):
    """Make batch predictions, clipped to zero."""
    if model_bundle['type'] == 'tabular':
        prediction = model_bundle['model'].predict(input_data)
        return np.clip(prediction, a_min=0, a_max=None)
    raise ValueError("Batch transform via CSV/JSON tabular input is only supported for "
                     "classical (tabular) models; sequence models require pre-built windows.")


def output_fn(prediction, accept):
    """Format batch prediction output as one value per line (CSV) or a JSON list."""
    if accept == 'text/csv':
        return '\n'.join(str(x) for x in prediction)
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    raise ValueError(f"Unsupported accept type: {accept}")
