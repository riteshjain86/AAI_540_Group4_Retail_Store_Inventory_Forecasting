"""
Forecast evaluation module.

Metrics follow the design document: MAE and RMSE for absolute error scale,
WAPE as the primary scale-independent metric (robust to the many zero/near-zero
demand days in retail data), and sMAPE as a supplementary metric.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error: sum of absolute errors / sum of actuals.

    Unlike MAPE, WAPE is well defined even when individual actuals are zero,
    which is common for low-volume store/product/day combinations.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Symmetric Mean Absolute Percentage Error, with an epsilon guard against 0/0 terms."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the full forecast-accuracy metric suite for one set of predictions.

    Args:
        y_true: Actual demand values
        y_pred: Forecast demand values (same length/order as y_true)

    Returns:
        Dictionary with mae, rmse, wape and smape
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'wape': wape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }


def evaluate_by_segment(y_true, y_pred, segment: np.ndarray) -> "pd.DataFrame":
    """
    Break down WAPE and MAE by a segment column (e.g. Category or Store ID).

    Useful because aggregate metrics can be dominated by high-volume
    store/product combinations, masking poor performance on smaller segments.

    Args:
        y_true: Actual demand values
        y_pred: Forecast demand values
        segment: Array-like of segment labels, same length as y_true

    Returns:
        DataFrame with one row per segment and its mae/wape/n_samples
    """
    import pandas as pd

    df = pd.DataFrame({'y_true': np.asarray(y_true, dtype=float),
                       'y_pred': np.asarray(y_pred, dtype=float),
                       'segment': segment})
    rows = []
    for seg, group in df.groupby('segment'):
        metrics = evaluate_forecast(group['y_true'], group['y_pred'])
        rows.append({'segment': seg, 'n_samples': len(group), **metrics})
    return pd.DataFrame(rows).sort_values('wape')
