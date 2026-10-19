"""
Tests for the SageMaker Pipeline (CI/CD DAG), its step scripts, the
monitoring reports and the infrastructure dashboard definition.
No AWS access is needed: S3 is mocked with moto where the SDK uploads code.
"""

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.test_data_preprocessing import make_raw_df

SAGEMAKER_DIR = Path(__file__).resolve().parents[2] / 'src' / 'aws' / 'sagemaker'
sys.path.insert(0, str(SAGEMAKER_DIR / 'scripts'))
sys.path.insert(0, str(SAGEMAKER_DIR / 'pipeline_scripts'))

import evaluate as evaluate_step  # noqa: E402
import preprocess as preprocess_step  # noqa: E402
import train as train_script  # noqa: E402


def write_split(path: Path, df: pd.DataFrame, name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / f'{name}.parquet', index=False)
    return str(path)


@pytest.fixture
def splits(tmp_path):
    df = make_raw_df(n_days=200, stores=('S001', 'S002', 'S003'))
    train, val, test = preprocess_step.chronological_split(df, 0.2, 0.2)
    return {name: write_split(tmp_path / name, part, name)
            for name, part in (('train', train), ('validation', val), ('test', test))}


# ---- train.py: test-set evaluation ------------------------------------

def test_evaluation_report_compares_against_seasonal_naive():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    report = train_script.build_evaluation_report('ridge', y_true, y_true * 1.1, y_true * 1.5, {'wape': 0.2})
    assert report['test']['wape'] == pytest.approx(0.1)
    assert report['baseline_seasonal_naive']['wape'] == pytest.approx(0.5)
    assert report['wape_improvement_vs_baseline'] == pytest.approx(0.8)
    assert report['regression_metrics']['wape']['value'] == pytest.approx(0.1)
    assert report['n_test_rows'] == 4


def test_ridge_training_writes_evaluation_into_model_dir(splits, tmp_path, monkeypatch):
    model_dir, output_dir = tmp_path / 'model', tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--model_type', 'ridge', '--train', splits['train'], '--validation', splits['validation'],
        '--test', splits['test'], '--model_dir', str(model_dir), '--output_dir', str(output_dir),
    ])
    train_script.main()

    report = json.loads((model_dir / 'evaluation.json').read_text())
    assert (model_dir / 'model.joblib').exists()
    assert (output_dir / 'evaluation.json').exists()
    assert set(report['test']) == {'mae', 'rmse', 'wape'}
    assert report['n_test_rows'] > 0
    assert report['model_type'] == 'ridge'


def test_training_without_test_channel_skips_evaluation(splits, tmp_path, monkeypatch):
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(sys, 'argv', [
        'train.py', '--model_type', 'ridge', '--train', splits['train'], '--validation', splits['validation'],
        '--model_dir', str(model_dir), '--output_dir', str(tmp_path / 'output'),
    ])
    monkeypatch.delenv('SM_CHANNEL_TEST', raising=False)
    train_script.main()
    assert not (model_dir / 'evaluation.json').exists()


def test_build_sequences_index_points_at_target_rows():
    df, feature_cols = train_script.engineer_features(make_raw_df(n_days=60))
    X, y, rows = train_script.build_sequences(df, feature_cols, 5, return_index=True)
    assert len(X) == len(y) == len(rows)
    np.testing.assert_array_equal(df[train_script.TARGET_COLUMN].to_numpy()[rows], y)


# ---- pipeline step scripts --------------------------------------------

def test_chronological_split_is_disjoint_and_ordered():
    df = make_raw_df(n_days=100)
    train, val, test = preprocess_step.chronological_split(df, 0.15, 0.15)
    assert len(train) + len(val) + len(test) == len(df)
    assert train['Date'].max() < val['Date'].min()
    assert val['Date'].max() < test['Date'].min()


def make_model_tar(path: Path, members: dict) -> str:
    with tarfile.open(path, 'w:gz') as tar:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_evaluate_step_reads_report_without_loading_model(tmp_path):
    report = {'test': {'wape': 0.6}, 'baseline_seasonal_naive': {'wape': 0.8}, 'wape_improvement_vs_baseline': 0.25}
    tar = make_model_tar(tmp_path / 'model.tar.gz', {
        'model.joblib': 'not a real pickle', 'evaluation.json': json.dumps(report)})
    assert evaluate_step.read_evaluation(tar) == report


def test_evaluate_step_rejects_artifact_without_report(tmp_path):
    tar = make_model_tar(tmp_path / 'model.tar.gz', {'model.joblib': 'x'})
    with pytest.raises(FileNotFoundError):
        evaluate_step.read_evaluation(tar)


def test_evaluate_step_rejects_incomplete_report(tmp_path):
    tar = make_model_tar(tmp_path / 'model.tar.gz', {'evaluation.json': json.dumps({'test': {}})})
    with pytest.raises(ValueError):
        evaluate_step.read_evaluation(tar)


# ---- pipeline definition ----------------------------------------------

def test_pipeline_dag_has_quality_gate_with_register_and_fail_branches(monkeypatch):
    moto = pytest.importorskip('moto')
    import boto3

    for key, value in {'AWS_ACCESS_KEY_ID': 'testing', 'AWS_SECRET_ACCESS_KEY': 'testing',
                       'AWS_DEFAULT_REGION': 'us-east-1'}.items():
        monkeypatch.setenv(key, value)
    bucket = 'sagemaker-us-east-1-123456789012'
    with moto.mock_aws():
        boto3.client('s3', region_name='us-east-1').create_bucket(Bucket=bucket)
        from sagemaker.workflow.pipeline_context import PipelineSession
        from src.aws.sagemaker import pipeline as pipeline_module

        session = PipelineSession(boto_session=boto3.session.Session(region_name='us-east-1'), default_bucket=bucket)
        definition = json.loads(pipeline_module.build_pipeline(
            'arn:aws:iam::123456789012:role/test', bucket, 'us-east-1', session).definition())

    steps = {s['Name']: s for s in definition['Steps']}
    assert list(steps) == ['Preprocess', 'TrainModel', 'EvaluateModel', 'CheckQualityGate']
    gate = steps['CheckQualityGate']['Arguments']
    assert [s['Type'] for s in gate['IfSteps']] == ['RegisterModel']
    assert [s['Type'] for s in gate['ElseSteps']] == ['Fail']
    paths = {c['LeftValue']['Std:JsonGet']['Path'] for c in gate['Conditions']}
    assert paths == {'test.wape', 'wape_improvement_vs_baseline'}
    defaults = {p['Name']: p.get('DefaultValue') for p in definition['Parameters']}
    assert defaults['MaxTestWAPE'] == pipeline_module.DEFAULT_MAX_TEST_WAPE
    assert defaults['MinWAPEImprovement'] == pipeline_module.DEFAULT_MIN_WAPE_IMPROVEMENT
    assert set(steps['TrainModel']['Arguments']['InputDataConfig'][i]['ChannelName'] for i in range(3)) == \
        {'train', 'validation', 'test'}


# ---- monitoring reports -----------------------------------------------

def test_model_quality_report_overall_and_segments():
    from src.aws.sagemaker import monitoring_reports as mr

    features = mr.batch_feature_frame(make_raw_df(n_days=60))
    predictions = features[train_script.TARGET_COLUMN].to_numpy() + 5.0
    report = mr.build_model_quality_report(features, predictions)
    assert report['rows'] == len(features)
    assert report['overall']['mae'] == pytest.approx(5.0)
    assert set(report['segments']) == set(mr.SEGMENT_COLS)
    assert sum(r['rows'] for r in report['segments']['Store ID']) == len(features)
    assert '| Deployed model (batch transform) |' in mr.model_quality_markdown(report)


def test_model_quality_report_rejects_misaligned_predictions():
    from src.aws.sagemaker import monitoring_reports as mr

    features = mr.batch_feature_frame(make_raw_df(n_days=60))
    with pytest.raises(ValueError):
        mr.build_model_quality_report(features, np.zeros(len(features) - 1))


def test_psi_flags_shifted_distribution_only():
    from src.aws.sagemaker import monitoring_reports as mr

    rng = np.random.default_rng(0)
    base = rng.normal(100, 10, 5000)
    assert mr.population_stability_index(base, rng.normal(100, 10, 5000)) < 0.1
    assert mr.population_stability_index(base, rng.normal(130, 10, 5000)) > mr.PSI_ALERT


def test_violation_summary_and_column_sanitizing():
    from src.aws.sagemaker import monitoring_reports as mr

    table = mr.summarize_violations({'violations': [
        {'feature_name': 'month', 'constraint_check_type': 'baseline_drift_check', 'description': 'drift'}]})
    assert table.to_dict(orient='records') == [
        {'feature': 'month', 'check': 'baseline_drift_check', 'description': 'drift'}]
    assert mr.summarize_violations({}).empty
    assert mr.sanitize_column('Holiday/Promotion') == 'Holiday_Promotion'
    assert mr.sanitize_column('Units Sold_lag_7') == 'Units_Sold_lag_7'


# ---- infrastructure dashboard -----------------------------------------

def test_dashboard_uses_real_dimensions(monkeypatch):
    from src.aws.sagemaker import infrastructure_monitoring as im

    monkeypatch.setattr(im.boto3, 'client', lambda *a, **k: SimpleNamespace())
    body = im.InfrastructureMonitoring().build_comprehensive_dashboard('sagemaker-us-east-1-123456789012')
    text = json.dumps(body)
    assert '"*"' not in text
    assert 'AWS/Sagemaker/ModelBuildingPipeline' in text
    assert '/aws/sagemaker/TransformJobs' in text
    assert 'sagemaker-us-east-1-123456789012' in text
    for w in body['widgets']:
        assert w['x'] + w['width'] <= 24
