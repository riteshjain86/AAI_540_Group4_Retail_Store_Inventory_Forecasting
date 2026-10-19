"""Data loaders and transformations used by the Streamlit dashboard."""

from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import pandas as pd

MODEL_COLUMNS = ['model', 'run_name', 'val_mae', 'val_rmse', 'val_wape', 'val_smape', 'run_id']
LOSS_COLUMNS = ['step', 'train_loss', 'val_loss']
SUMMARY_METRICS = ('test_', 'test_naive_')


def _empty_model_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MODEL_COLUMNS)


def _display_model_name(run_name: str) -> str:
    if not run_name.startswith('classical__'):
        return run_name
    model_name = run_name.removeprefix('classical__')
    if model_name.endswith('__tuned'):
        return f'{model_name.removesuffix("__tuned")} (tuned)'
    return model_name


def _parse_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == 'true'


def load_model_comparison(tracking_uri: str, experiment_name: str) -> tuple[pd.DataFrame, dict | None]:
    """Load the latest pipeline run and its nested validation runs."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return _empty_model_frame(), None

    parents = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'training_pipeline'",
        order_by=['attribute.start_time DESC'],
        max_results=1,
    )
    if not parents:
        return _empty_model_frame(), None

    parent = parents[0]
    children = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    rows = []
    for child in children:
        run_name = child.data.tags.get('mlflow.runName', '')
        row = {
            'model': _display_model_name(run_name),
            'run_name': run_name,
            'run_id': child.info.run_id,
        }
        row.update({
            metric: child.data.metrics.get(metric, np.nan)
            for metric in ('val_mae', 'val_rmse', 'val_wape', 'val_smape')
        })
        rows.append(row)

    models = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    if not models.empty:
        models = models.sort_values('val_wape', ascending=True, na_position='last').reset_index(drop=True)

    summary = {
        'run_id': parent.info.run_id,
        'start_time': parent.info.start_time,
        'selected_model': parent.data.params.get('selected_model'),
        'model_type': parent.data.params.get('model_type'),
        'quality_gates_passed': _parse_bool(parent.data.tags.get('quality_gates_passed')),
    }
    summary.update({
        key: value for key, value in parent.data.metrics.items()
        if key.startswith(SUMMARY_METRICS) or key == 'mae_improvement_over_naive'
    })
    return models, summary


def load_loss_curves(tracking_uri: str, run_id: str) -> pd.DataFrame:
    """Return per-step training and validation losses for one MLflow run."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    histories = {
        name: {metric.step: metric.value for metric in client.get_metric_history(run_id, name)}
        for name in ('train_loss', 'val_loss')
    }
    steps = sorted(set(histories['train_loss']) | set(histories['val_loss']))
    return pd.DataFrame(
        [
            {
                'step': step,
                'train_loss': histories['train_loss'].get(step, np.nan),
                'val_loss': histories['val_loss'].get(step, np.nan),
            }
            for step in steps
        ],
        columns=LOSS_COLUMNS,
    )


def load_batch_predictions(source: str) -> np.ndarray:
    """Load a headerless, single-column batch-transform CSV from disk or S3."""
    if source.startswith('s3://'):
        parsed = urlparse(source)
        if not parsed.netloc or not parsed.path.lstrip('/'):
            raise ValueError(f'Invalid S3 predictions URI: {source}')
        import boto3

        body = boto3.client('s3').get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip('/'),
        )['Body'].read()
        frame = pd.read_csv(BytesIO(body), header=None)
    else:
        frame = pd.read_csv(source, header=None)

    if frame.shape[1] != 1:
        raise ValueError(f'Expected a headerless single-column predictions CSV, found {frame.shape[1]} columns')
    return pd.to_numeric(frame.iloc[:, 0], errors='raise').to_numpy(dtype=float)


def predictions_vs_actuals(test_df: pd.DataFrame, predictions) -> pd.DataFrame:
    """Align predictions with the monitoring feature pipeline and actuals."""
    from src.aws.sagemaker.monitoring_reports import (
        SEASONAL_NAIVE_COL,
        TARGET_COLUMN,
        batch_feature_frame,
        build_model_quality_report,
    )

    features = batch_feature_frame(test_df)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    quality_report = build_model_quality_report(features, predictions)
    result = pd.DataFrame({
        'Date': features['Date'].to_numpy(),
        'Store ID': features['Store ID'].to_numpy(),
        'Product ID': features['Product ID'].to_numpy(),
        'actual': features[TARGET_COLUMN].to_numpy(),
        'predicted': predictions,
        'seasonal_naive': features[SEASONAL_NAIVE_COL].to_numpy(),
    })
    result['abs_error'] = (result['actual'] - result['predicted']).abs()
    result.attrs['quality_report'] = quality_report
    return result
