import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.ui.dashboard_data import (
    load_batch_predictions,
    load_loss_curves,
    load_model_comparison,
    predictions_vs_actuals,
)
from tests.test_data_preprocessing import make_raw_df


def _create_run(client, experiment_id, run_name, start_time, parent_id=None, params=None, metrics=None):
    tags = {'mlflow.runName': run_name}
    if parent_id is not None:
        tags['mlflow.parentRunId'] = parent_id
    run = client.create_run(experiment_id, start_time=start_time, tags=tags)
    for key, value in (params or {}).items():
        client.log_param(run.info.run_id, key, value)
    for key, value in (metrics or {}).items():
        client.log_metric(run.info.run_id, key, value, timestamp=start_time, step=0)
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def _create_tracking_store(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment('retail_inventory_forecasting')
    return tracking_uri, client, experiment_id


def test_load_model_comparison_uses_latest_parent_and_sorts_children(tmp_path):
    tracking_uri, client, experiment_id = _create_tracking_store(tmp_path)
    old_parent_id = _create_run(client, experiment_id, 'training_pipeline', 1000)
    _create_run(
        client, experiment_id, 'classical__ridge', 1100, old_parent_id,
        metrics={'val_wape': 0.01},
    )

    parent_id = _create_run(
        client,
        experiment_id,
        'training_pipeline',
        2000,
        params={'selected_model': 'classical', 'model_type': 'ridge'},
        metrics={
            'test_mae': 10.0,
            'test_rmse': 15.0,
            'test_wape': 0.2,
            'test_smape': 0.3,
            'test_naive_mae': 20.0,
            'test_naive_wape': 0.4,
            'mae_improvement_over_naive': 0.5,
        },
    )
    client.set_tag(parent_id, 'quality_gates_passed', 'True')
    _create_run(
        client, experiment_id, 'seasonal_naive', 2100, parent_id,
        metrics={'val_mae': 20, 'val_rmse': 25, 'val_wape': 0.8, 'val_smape': 0.9},
    )
    _create_run(
        client, experiment_id, 'classical__ridge', 2200, parent_id,
        metrics={'val_mae': 11, 'val_rmse': 16, 'val_wape': 0.5, 'val_smape': 0.6},
    )
    tuned_id = _create_run(
        client, experiment_id, 'classical__ridge__tuned', 2300, parent_id,
        metrics={'val_mae': 9, 'val_rmse': 14, 'val_wape': 0.4, 'val_smape': 0.5},
    )
    lstm_id = _create_run(
        client, experiment_id, 'lstm', 2400, parent_id,
        metrics={'val_mae': 12, 'val_rmse': 17, 'val_wape': 0.45, 'val_smape': 0.55},
    )
    for step, (train_loss, val_loss) in enumerate(((0.9, 1.1), (0.7, 0.8), (0.5, 0.6))):
        client.log_metric(lstm_id, 'train_loss', train_loss, timestamp=2500 + step, step=step)
        client.log_metric(lstm_id, 'val_loss', val_loss, timestamp=2500 + step, step=step)

    models, summary = load_model_comparison(tracking_uri, 'retail_inventory_forecasting')

    assert list(models.columns) == [
        'model', 'run_name', 'val_mae', 'val_rmse', 'val_wape', 'val_smape', 'run_id',
    ]
    assert list(models['model']) == ['ridge (tuned)', 'lstm', 'ridge', 'seasonal_naive']
    assert models.iloc[0]['run_id'] == tuned_id
    assert summary == {
        'run_id': parent_id,
        'start_time': 2000,
        'selected_model': 'classical',
        'model_type': 'ridge',
        'quality_gates_passed': True,
        'test_mae': 10.0,
        'test_rmse': 15.0,
        'test_wape': 0.2,
        'test_smape': 0.3,
        'test_naive_mae': 20.0,
        'test_naive_wape': 0.4,
        'mae_improvement_over_naive': 0.5,
    }

    losses = load_loss_curves(tracking_uri, lstm_id)
    assert list(losses.columns) == ['step', 'train_loss', 'val_loss']
    assert losses.to_dict(orient='records') == [
        {'step': 0, 'train_loss': 0.9, 'val_loss': 1.1},
        {'step': 1, 'train_loss': 0.7, 'val_loss': 0.8},
        {'step': 2, 'train_loss': 0.5, 'val_loss': 0.6},
    ]


def test_load_model_comparison_returns_empty_for_experiment_without_runs(tmp_path):
    tracking_uri, client, _ = _create_tracking_store(tmp_path)
    client.create_experiment('empty_experiment')

    models, summary = load_model_comparison(tracking_uri, 'empty_experiment')

    assert models.empty
    assert list(models.columns) == [
        'model', 'run_name', 'val_mae', 'val_rmse', 'val_wape', 'val_smape', 'run_id',
    ]
    assert summary is None


def test_load_batch_predictions_reads_headerless_local_csv(tmp_path):
    predictions_path = tmp_path / 'predictions.csv'
    predictions_path.write_text('1.25\n2.5\n3.75\n')

    predictions = load_batch_predictions(str(predictions_path))

    np.testing.assert_array_equal(predictions, np.array([1.25, 2.5, 3.75]))


def test_predictions_vs_actuals_aligns_rows_and_rejects_mismatch():
    test_df = make_raw_df(n_days=60)
    from src.aws.sagemaker.monitoring_reports import batch_feature_frame

    features = batch_feature_frame(test_df)
    predictions = features['Units Sold'].to_numpy() + 5.0

    result = predictions_vs_actuals(test_df, predictions)

    assert list(result.columns) == [
        'Date', 'Store ID', 'Product ID', 'actual', 'predicted', 'seasonal_naive', 'abs_error',
    ]
    assert len(result) == len(features)
    assert (result['abs_error'] == 5.0).all()
    assert result.attrs['quality_report']['overall']['mae'] == pytest.approx(5.0)
    with pytest.raises(ValueError, match='predictions but .* feature rows'):
        predictions_vs_actuals(test_df, predictions[:-1])
