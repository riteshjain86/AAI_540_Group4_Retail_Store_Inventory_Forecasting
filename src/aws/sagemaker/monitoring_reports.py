"""
Model- and data-monitoring reports for the retail demand forecaster.

Two reports, both written under reports/monitoring/ locally and uploaded to
s3://<bucket>/aai540/retail-forecasting/monitoring/:

  data-quality   SageMaker Model Monitor data-quality baseline (statistics +
                 constraints) on the engineered training features, then an
                 on-demand Model Monitor analysis of the batch-inference input
                 (test-period features) against that baseline, which writes
                 constraint_violations.json. A PSI drift table is added so the
                 report is readable without opening the raw JSON.

  model-quality  Joins the batch transform output (predictions/batch_input.csv.out)
                 with the actual Units Sold for the same rows and reports
                 MAE / RMSE / WAPE overall and per store, category and region,
                 next to the seasonal-naive baseline. Metrics are published to
                 CloudWatch (RetailForecasting/ModelQuality) for the dashboards.

Usage (from SageMaker Studio):
    python -m src.aws.sagemaker.monitoring_reports model-quality
    python -m src.aws.sagemaker.monitoring_reports data-quality
    python -m src.aws.sagemaker.monitoring_reports all
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PREFIX = 'aai540/retail-forecasting'
SCRIPTS_DIR = Path(__file__).parent / 'scripts'
REPORT_DIR = Path('reports/monitoring')
SEGMENT_COLS = ['Store ID', 'Category', 'Region']
PSI_ALERT = 0.2


def parse_batch_output(body: str) -> np.ndarray:
    """Parse batch predictions from JSON or comma/newline-delimited text."""
    stripped = body.strip()
    if stripped.startswith('{'):
        values = json.loads(stripped)['predictions']
    else:
        values = [token.strip() for token in re.split(r'[,\r\n]+', stripped) if token.strip()]
    try:
        return np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError('Batch output contains a non-numeric prediction') from exc


sys.path.insert(0, str(SCRIPTS_DIR))
from train import (CATEGORICAL_COLS, NUMERIC_COLS,  # noqa: E402
                   TARGET_COLUMN, engineer_features, evaluate)

SEASONAL_NAIVE_COL = f'{TARGET_COLUMN}_lag_7'


def sanitize_column(name: str) -> str:
    """Model Monitor column names: alphanumerics and underscores only."""
    return re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_')


def batch_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features exactly as batch_inference.prepare_batch_input() does,
    keeping the target, so row i lines up with line i of the transform output.
    """
    engineered, _ = engineer_features(raw_df.copy())
    return engineered[NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COLUMN, 'Date']]


def population_stability_index(baseline, current, bins: int = 10) -> float:
    baseline, current = np.asarray(baseline, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(current, baseline[0]) else float('inf')
    edges[0], edges[-1] = -np.inf, np.inf
    base_pct = np.histogram(baseline, edges)[0] / len(baseline)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    base_pct, cur_pct = np.clip(base_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct)))


def drift_table(baseline: pd.DataFrame, current: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        psi = population_stability_index(baseline[col], current[col])
        rows.append({
            'feature': col,
            'baseline_mean': float(baseline[col].mean()),
            'current_mean': float(current[col].mean()),
            'psi': psi,
            'drift': 'ALERT' if psi >= PSI_ALERT else ('watch' if psi >= 0.1 else 'ok'),
        })
    return pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)


def segment_metrics(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    rows = []
    for value, part in df.groupby(segment_col):
        m = evaluate(part[TARGET_COLUMN], part['prediction'])
        rows.append({segment_col: value, 'rows': len(part), **m,
                     'baseline_wape': evaluate(part[TARGET_COLUMN], part[SEASONAL_NAIVE_COL])['wape']})
    return pd.DataFrame(rows).sort_values('wape').reset_index(drop=True)


def build_model_quality_report(features: pd.DataFrame, predictions) -> dict:
    """Pure computation of the model-quality report (no AWS calls)."""
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(predictions) != len(features):
        raise ValueError(f'{len(predictions)} predictions but {len(features)} feature rows; '
                         'was the batch input built from a different test split?')
    df = features.copy()
    df['prediction'] = predictions
    overall = evaluate(df[TARGET_COLUMN], df['prediction'])
    baseline = evaluate(df[TARGET_COLUMN], df[SEASONAL_NAIVE_COL])
    daily = df.groupby('Date')[[TARGET_COLUMN, 'prediction']].sum().reset_index()
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'rows': int(len(df)),
        'date_range': [str(pd.to_datetime(df['Date']).min().date()), str(pd.to_datetime(df['Date']).max().date())],
        'prediction_stats': {
            'min': float(predictions.min()), 'mean': float(predictions.mean()), 'max': float(predictions.max()),
            'negative': int((predictions < 0).sum()), 'nan': int(np.isnan(predictions).sum()),
        },
        'overall': overall,
        'baseline_seasonal_naive': baseline,
        'wape_improvement_vs_baseline': 1.0 - overall['wape'] / baseline['wape'] if baseline['wape'] else 0.0,
        'segments': {col: segment_metrics(df, col).to_dict(orient='records') for col in SEGMENT_COLS},
        'daily_totals': daily.assign(Date=daily['Date'].astype(str)).to_dict(orient='records'),
    }


def model_quality_markdown(report: dict) -> str:
    o, b = report['overall'], report['baseline_seasonal_naive']
    lines = [
        '# Model quality report: batch inference vs actuals',
        '',
        f"Generated {report['generated_at']} | {report['rows']} forecasts | "
        f"{report['date_range'][0]} to {report['date_range'][1]}",
        '',
        '| Model | MAE | RMSE | WAPE |',
        '|---|---|---|---|',
        f"| Deployed model (batch transform) | {o['mae']:.2f} | {o['rmse']:.2f} | {o['wape']:.3f} |",
        f"| Seasonal-naive baseline | {b['mae']:.2f} | {b['rmse']:.2f} | {b['wape']:.3f} |",
        '',
        f"WAPE improvement vs baseline: {report['wape_improvement_vs_baseline']:.1%}. "
        f"Predictions: min {report['prediction_stats']['min']:.1f}, mean {report['prediction_stats']['mean']:.1f}, "
        f"max {report['prediction_stats']['max']:.1f}, negative {report['prediction_stats']['negative']}, "
        f"NaN {report['prediction_stats']['nan']}.",
    ]
    for col, rows in report['segments'].items():
        lines += ['', f'## By {col}', '', f'| {col} | Rows | MAE | WAPE | Baseline WAPE |', '|---|---|---|---|---|']
        lines += [f"| {r[col]} | {r['rows']} | {r['mae']:.2f} | {r['wape']:.3f} | {r['baseline_wape']:.3f} |" for r in rows]
    return '\n'.join(lines) + '\n'


def summarize_violations(violations: dict) -> pd.DataFrame:
    rows = [{'feature': v.get('feature_name'), 'check': v.get('constraint_check_type'),
             'description': v.get('description')} for v in violations.get('violations', [])]
    return pd.DataFrame(rows, columns=['feature', 'check', 'description'])


def data_quality_markdown(drift: pd.DataFrame, violations: pd.DataFrame, s3_paths: dict) -> str:
    lines = [
        '# Data quality report: batch-inference input vs training baseline',
        '',
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        '',
        f"- Baseline (Model Monitor statistics/constraints): `{s3_paths['baseline']}`",
        f"- Drift analysis output (constraint_violations.json): `{s3_paths['analysis']}`",
        '',
        f'## SageMaker Model Monitor constraint violations ({len(violations)})',
        '',
    ]
    if violations.empty:
        lines.append('No constraint violations.')
    else:
        lines += ['| Feature | Check | Description |', '|---|---|---|']
        lines += [f"| {r.feature} | {r.check} | {r.description} |" for r in violations.itertuples()]
    lines += ['', f'## Feature drift (PSI, alert >= {PSI_ALERT})', '',
              '| Feature | Train mean | Batch mean | PSI | Status |', '|---|---|---|---|---|']
    lines += [f"| {r.feature} | {r.baseline_mean:.2f} | {r.current_mean:.2f} | {r.psi:.3f} | {r.drift} |"
              for r in drift.itertuples()]
    return '\n'.join(lines) + '\n'


class MonitoringReports:
    """Generate the monitoring reports against the project's S3 layout."""

    def __init__(self):
        load_dotenv()
        import sagemaker

        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.boto_session = boto3.session.Session(region_name=self.region)
        account = os.getenv('AWS_ACCOUNT_ID') or self.boto_session.client('sts').get_caller_identity()['Account']
        self.bucket = f'sagemaker-{self.region}-{account}'
        self.s3 = self.boto_session.client('s3')
        self.sagemaker_session = sagemaker.Session(self.boto_session)
        self.role = os.getenv('AWS_ROLE_ARN') or sagemaker.get_execution_role(sagemaker_session=self.sagemaker_session)
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def s3_uri(self, key: str) -> str:
        return f's3://{self.bucket}/{key}'

    def load_split(self, split: str) -> pd.DataFrame:
        local = Path(f'data/processed/{split}.parquet')
        if local.exists():
            return pd.read_parquet(local)
        logger.info(f'{local} not found; reading {split} split from S3')
        obj = self.s3.get_object(Bucket=self.bucket, Key=f'{BASE_PREFIX}/{split}/{split}.parquet')
        from io import BytesIO
        return pd.read_parquet(BytesIO(obj['Body'].read()))

    def upload(self, local_path: Path, key: str) -> str:
        self.s3.upload_file(str(local_path), self.bucket, key)
        return self.s3_uri(key)

    @staticmethod
    def _parse_batch_output(body: str) -> np.ndarray:
        """Parse batch-transform output as JSON or comma/newline-delimited predictions."""
        return parse_batch_output(body)

    # ---- model quality -------------------------------------------------
    def model_quality(self, output_key: str = f'{BASE_PREFIX}/predictions/batch_input.csv.out',
                      publish: bool = True) -> dict:
        body = self.s3.get_object(Bucket=self.bucket, Key=output_key)['Body'].read().decode('utf-8')
        predictions = np.asarray(self._parse_batch_output(body), dtype=float)
        report = build_model_quality_report(batch_feature_frame(self.load_split('test')), predictions)
        report['predictions_s3_uri'] = self.s3_uri(output_key)

        json_path, md_path = REPORT_DIR / 'model_quality.json', REPORT_DIR / 'model_quality.md'
        json_path.write_text(json.dumps(report, indent=2, default=str))
        md_path.write_text(model_quality_markdown(report))
        self._plot_model_quality(report)
        prefix = f'{BASE_PREFIX}/monitoring/model-quality/{self.run_id}'
        for path in REPORT_DIR.glob('model_quality*'):
            self.upload(path, f'{prefix}/{path.name}')

        if publish:
            from src.aws.sagemaker.monitoring import SageMakerModelMonitor
            monitor = SageMakerModelMonitor()
            monitor.publish_forecast_accuracy_metrics(report['overall'], segment='overall')
            monitor.publish_forecast_accuracy_metrics(report['baseline_seasonal_naive'], segment='seasonal_naive')
            for row in report['segments']['Store ID']:
                monitor.publish_forecast_accuracy_metrics(
                    {k: row[k] for k in ('mae', 'rmse', 'wape')}, segment=f"store_{row['Store ID']}")

        logger.info(f"Model quality: {report['overall']} (baseline {report['baseline_seasonal_naive']}); "
                    f"report at {md_path} and {self.s3_uri(prefix)}/")
        return report

    @staticmethod
    def _plot_model_quality(report: dict):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        daily = pd.DataFrame(report['daily_totals'])
        stores = pd.DataFrame(report['segments']['Store ID'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        axes[0].plot(pd.to_datetime(daily['Date']), daily[TARGET_COLUMN], label='Actual')
        axes[0].plot(pd.to_datetime(daily['Date']), daily['prediction'], label='Forecast')
        axes[0].set_title('Daily total units: actual vs batch forecast')
        axes[0].legend()
        axes[1].bar(stores['Store ID'].astype(str), stores['wape'], label='Model')
        axes[1].plot(stores['Store ID'].astype(str), stores['baseline_wape'], 'ro', label='Seasonal-naive')
        axes[1].set_title('WAPE by store')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(REPORT_DIR / 'model_quality.png', dpi=120)
        plt.close(fig)

    # ---- data quality --------------------------------------------------
    def _write_monitor_csv(self, df: pd.DataFrame, name: str) -> str:
        frame = df[NUMERIC_COLS + CATEGORICAL_COLS].rename(columns=sanitize_column)
        local = REPORT_DIR / f'{name}.csv'
        frame.to_csv(local, index=False)
        return self.upload(local, f'{BASE_PREFIX}/monitoring/data-quality/{name}/{name}.csv')

    def data_quality(self, instance_type: str = 'ml.m5.xlarge') -> dict:
        from sagemaker import image_uris
        from sagemaker.model_monitor import DefaultModelMonitor
        from sagemaker.model_monitor.dataset_format import DatasetFormat
        from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

        baseline_df = batch_feature_frame(self.load_split('train'))
        current_df = batch_feature_frame(self.load_split('test'))
        baseline_uri = self._write_monitor_csv(baseline_df, 'baseline-input')
        current_uri = self._write_monitor_csv(current_df, 'batch-input')

        prefix = f'{BASE_PREFIX}/monitoring/data-quality'
        baseline_out = self.s3_uri(f'{prefix}/baseline')
        analysis_out = self.s3_uri(f'{prefix}/analysis/{self.run_id}')

        logger.info('Running Model Monitor suggest_baseline on training features...')
        monitor = DefaultModelMonitor(role=self.role, instance_count=1, instance_type=instance_type,
                                      volume_size_in_gb=20, max_runtime_in_seconds=1800,
                                      sagemaker_session=self.sagemaker_session)
        monitor.suggest_baseline(baseline_dataset=baseline_uri, dataset_format=DatasetFormat.csv(header=True),
                                 output_s3_uri=baseline_out, wait=True, logs=False,
                                 job_name=f'retail-dq-baseline-{self.run_id}')

        logger.info('Running on-demand Model Monitor analysis of batch input against the baseline...')
        analyzer = Processor(
            role=self.role, image_uri=image_uris.retrieve('model-monitor', self.region),
            instance_count=1, instance_type=instance_type, volume_size_in_gb=20,
            max_runtime_in_seconds=1800, sagemaker_session=self.sagemaker_session,
            env={
                'dataset_format': json.dumps(DatasetFormat.csv(header=True)),
                'dataset_source': '/opt/ml/processing/input/data',
                'output_path': '/opt/ml/processing/output',
                'publish_cloudwatch_metrics': 'Disabled',
                'baseline_statistics': '/opt/ml/processing/baseline/statistics/statistics.json',
                'baseline_constraints': '/opt/ml/processing/baseline/constraints/constraints.json',
            },
        )
        analyzer.run(
            inputs=[
                ProcessingInput(source=current_uri, destination='/opt/ml/processing/input/data'),
                ProcessingInput(source=f'{baseline_out}/statistics.json',
                                destination='/opt/ml/processing/baseline/statistics'),
                ProcessingInput(source=f'{baseline_out}/constraints.json',
                                destination='/opt/ml/processing/baseline/constraints'),
            ],
            outputs=[ProcessingOutput(source='/opt/ml/processing/output', destination=analysis_out)],
            job_name=f'retail-dq-analysis-{self.run_id}', wait=True, logs=False,
        )

        violations = {}
        try:
            body = self.s3.get_object(Bucket=self.bucket,
                                      Key=f'{prefix}/analysis/{self.run_id}/constraint_violations.json')['Body']
            violations = json.loads(body.read())
        except self.s3.exceptions.NoSuchKey:
            logger.info('No constraint_violations.json produced (no violations).')

        numeric = [sanitize_column(c) for c in NUMERIC_COLS]
        drift = drift_table(baseline_df[NUMERIC_COLS].rename(columns=sanitize_column),
                            current_df[NUMERIC_COLS].rename(columns=sanitize_column), numeric)
        violations_df = summarize_violations(violations)
        md_path = REPORT_DIR / 'data_quality.md'
        md_path.write_text(data_quality_markdown(drift, violations_df,
                                                 {'baseline': baseline_out, 'analysis': analysis_out}))
        (REPORT_DIR / 'data_quality_violations.json').write_text(json.dumps(violations, indent=2))
        drift.to_csv(REPORT_DIR / 'data_quality_drift.csv', index=False)
        for name in ('data_quality.md', 'data_quality_violations.json', 'data_quality_drift.csv'):
            self.upload(REPORT_DIR / name, f'{prefix}/analysis/{self.run_id}/report/{name}')

        logger.info(f'{len(violations_df)} constraint violations; report at {md_path} and {analysis_out}/report/')
        return {'violations': violations_df, 'drift': drift, 'baseline': baseline_out, 'analysis': analysis_out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report', choices=['model-quality', 'data-quality', 'all'])
    parser.add_argument('--output-key', default=f'{BASE_PREFIX}/predictions/batch_input.csv.out',
                        help='S3 key of the batch transform output to evaluate')
    parser.add_argument('--no-publish', action='store_true', help='Skip publishing metrics to CloudWatch')
    args = parser.parse_args()

    reports = MonitoringReports()
    if args.report in ('model-quality', 'all'):
        reports.model_quality(output_key=args.output_key, publish=not args.no_publish)
    if args.report in ('data-quality', 'all'):
        reports.data_quality()


if __name__ == '__main__':
    main()
