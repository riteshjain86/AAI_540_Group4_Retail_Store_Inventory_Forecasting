import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_data_preprocessing import make_raw_df

SAGEMAKER_DIR = Path(__file__).resolve().parents[2] / 'src' / 'aws' / 'sagemaker'
sys.path.insert(0, str(SAGEMAKER_DIR / 'scripts'))

import train as train_script  # noqa: E402


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


def test_parse_batch_output_handles_csv_and_json():
    from src.aws.sagemaker import monitoring_reports as mr

    assert list(mr.MonitoringReports._parse_batch_output('1.5\n2.5\n3.5')) == [1.5, 2.5, 3.5]
    assert list(mr.MonitoringReports._parse_batch_output('1.5,2.5,3.5')) == [1.5, 2.5, 3.5]
    assert list(mr.MonitoringReports._parse_batch_output('1.5,2.5\n3.5\n')) == [1.5, 2.5, 3.5]
    assert list(mr.MonitoringReports._parse_batch_output('{"predictions": [1.5, 2.5, 3.5]}')) == [
        1.5, 2.5, 3.5,
    ]
    with pytest.raises(ValueError, match='non-numeric'):
        mr.MonitoringReports._parse_batch_output('1.5,not-a-number,3.5')


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


def test_search_metrics_are_wrapped_as_arrays(monkeypatch):
    """Every entry in a CloudWatch widget's 'metrics' list must itself be an
    array (classic tuple, or a single-element array wrapping a metric-math
    expression object) - a bare dict/object fails PutDashboard validation."""
    from src.aws.sagemaker import infrastructure_monitoring as im

    monkeypatch.setattr(im.boto3, 'client', lambda *a, **k: SimpleNamespace())
    body = im.InfrastructureMonitoring().build_comprehensive_dashboard('sagemaker-us-east-1-123456789012')
    for w in body['widgets']:
        for m in w.get('properties', {}).get('metrics', []):
            assert isinstance(m, list), f"metrics entry must be an array, got {type(m)}: {m}"
