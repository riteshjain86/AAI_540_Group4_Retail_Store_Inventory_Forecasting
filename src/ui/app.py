"""
Streamlit app for interactively testing the demand-forecasting model.

Two modes:
  - "Local model": loads models/model.keras (or model.joblib) directly and
    predicts in-process. Useful for quick sanity checks without any AWS
    dependency.
  - "SageMaker endpoint": sends the same feature window to a deployed
    real-time endpoint (see src/aws/sagemaker/deployment.py) and shows the
    raw response. Useful for demonstrating endpoint invocation.

Run with:
    streamlit run src/ui/app.py
from the repository root (after `pip install streamlit` / the full requirements.txt).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ui.dashboard_data import (  # noqa: E402
    load_batch_predictions,
    load_loss_curves,
    load_model_comparison,
    predictions_vs_actuals,
)
from src.data_ingestion.data_ingestion import (
    load_data, validate_schema, DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN,
)
from src.data_preprocessing.data_preprocessing import clean_data, clip_non_negative
from src.feature_engineering.feature_engineering import (
    add_calendar_features, add_lag_rolling_features, drop_lag_rolling_warmup,
    lag_rolling_feature_names,
)
from src.utils.helpers import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / 'data' / 'raw' / 'retail_store_inventory.csv'
MODEL_DIR = REPO_ROOT / 'models'
CONFIG_PATH = REPO_ROOT / 'config.yaml'
DEFAULT_PREDICTIONS_SOURCE = (
    's3://sagemaker-us-east-1-215595263640/'
    'aai540/retail-forecasting/predictions/batch_input.csv.out'
)


def resolve_tracking_uri(tracking_uri: str) -> str:
    if tracking_uri.startswith('sqlite:///'):
        database_path = Path(tracking_uri.removeprefix('sqlite:///'))
        if not database_path.is_absolute():
            database_path = REPO_ROOT / database_path
        return f'sqlite:///{database_path.resolve()}'
    return tracking_uri


@st.cache_data(ttl=30)
def cached_load_model_comparison(tracking_uri: str, experiment_name: str):
    return load_model_comparison(tracking_uri, experiment_name)


@st.cache_data
def cached_load_loss_curves(tracking_uri: str, run_id: str):
    return load_loss_curves(tracking_uri, run_id)


@st.cache_data
def cached_load_batch_predictions(source: str):
    return load_batch_predictions(source)


@st.cache_data
def cached_load_test_split(path: str):
    return pd.read_parquet(path)


def _selected_validation_model(summary: dict) -> str | None:
    selected_model = summary.get('selected_model')
    model_type = summary.get('model_type')
    if selected_model == 'classical' and model_type:
        return f'{model_type} (tuned)'
    return selected_model


def render_model_comparison(tracking_uri: str, experiment_name: str):
    try:
        models, summary = cached_load_model_comparison(tracking_uri, experiment_name)
    except Exception as exc:
        st.error(f'Could not load MLflow experiment: {exc}')
        return

    if summary is None:
        st.info(
            'No training pipeline run was found. From the repository root, run '
            '`python -m src.pipelines.training_pipeline` to train and log a model.'
        )
        return

    selected_model = _selected_validation_model(summary)
    selected_label = selected_model or summary.get('selected_model') or 'Not recorded'
    top_metrics = st.columns(2)
    top_metrics[0].metric('Selected model', selected_label)
    gate = summary.get('quality_gates_passed')
    gate_label = 'Passed' if gate is True else ('Failed' if gate is False else 'Not recorded')
    top_metrics[1].metric('Quality gates', gate_label)

    metric_names = ('mae', 'rmse', 'wape', 'smape')
    available_metrics = [
        metric for metric in metric_names
        if summary.get(f'test_{metric}') is not None
    ]
    if available_metrics:
        metric_columns = st.columns(len(available_metrics))
        for column, metric in zip(metric_columns, available_metrics):
            value = summary[f'test_{metric}']
            naive = summary.get(f'test_naive_{metric}')
            value_format = '{:.3f}' if metric in ('wape', 'smape') else '{:.2f}'
            delta = None if naive is None else f'{value - naive:+.3f} vs seasonal naive'
            column.metric(
                f'Test {metric.upper()}',
                value_format.format(value),
                delta=delta,
                delta_color='inverse',
            )

    start_time = summary.get('start_time')
    if start_time is not None:
        formatted_time = pd.to_datetime(start_time, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M UTC')
        st.caption(f"Parent run `{summary['run_id']}` · started {formatted_time}")

    if models.empty:
        st.info('The latest training pipeline run has no nested model-validation runs.')
        return

    st.subheader('Validation comparison')
    best_model = models.iloc[0]['model']

    def highlight_models(row):
        if row['model'] == selected_model:
            return ['background-color: #e6f4ea'] * len(row)
        if row['model'] == best_model:
            return ['background-color: #fff4cc'] * len(row)
        return [''] * len(row)

    styled_models = models.style.apply(highlight_models, axis=1).format({
        'val_mae': '{:.2f}',
        'val_rmse': '{:.2f}',
        'val_wape': '{:.4f}',
        'val_smape': '{:.4f}',
    }, na_rep='—')
    st.dataframe(styled_models, use_container_width=True, hide_index=True)

    st.subheader('Validation WAPE')
    st.bar_chart(models.set_index('model')[['val_wape']])

    for model in ('lstm', 'transformer'):
        matching_runs = models[models['model'] == model]
        if matching_runs.empty:
            continue
        losses = cached_load_loss_curves(tracking_uri, matching_runs.iloc[0]['run_id'])
        if not losses.empty:
            st.subheader(f'{model.upper()} training loss')
            st.line_chart(losses.set_index('step')[['train_loss', 'val_loss']])


def render_prediction_review():
    st.image(str(REPO_ROOT / 'docs' / 'architecture_diagram.png'), use_container_width=True)
    st.markdown(
        '1. **Feature Store** provides the retail data.  \n'
        '2. **SageMaker Pipeline** runs Preprocess → Train → Evaluate → WAPE gate.  \n'
        '3. Passing models are added to the **Model Registry**.  \n'
        '4. **Batch Transform** writes predictions to S3; an optional real-time endpoint '
        'serves individual requests.'
    )

    st.subheader('Batch prediction review')
    source_column, test_column = st.columns(2)
    with source_column:
        source = st.text_input('Predictions source (local path or S3 URI)', value=DEFAULT_PREDICTIONS_SOURCE)
    with test_column:
        test_path = st.text_input('Test split Parquet path', value='data/processed/test.parquet')

    if st.button('Load predictions'):
        try:
            resolved_test_path = Path(test_path)
            if not resolved_test_path.is_absolute():
                resolved_test_path = REPO_ROOT / resolved_test_path
            test_df = cached_load_test_split(str(resolved_test_path.resolve()))
            predictions = cached_load_batch_predictions(source.strip())
            review = predictions_vs_actuals(test_df, predictions)
            st.session_state['prediction_review'] = {
                'source': source.strip(),
                'test_path': test_path,
                'frame': review,
                'report': review.attrs['quality_report'],
            }
        except Exception as exc:
            st.session_state.pop('prediction_review', None)
            st.error(f'Could not load predictions: {exc}')

    loaded = st.session_state.get('prediction_review')
    if not loaded or loaded['source'] != source.strip() or loaded['test_path'] != test_path:
        st.info('Choose a predictions source and test split, then select “Load predictions”.')
        return

    report = loaded['report']
    model_metrics = report['overall']
    naive_metrics = report['baseline_seasonal_naive']
    st.caption(f"{report['rows']:,} aligned prediction rows · model compared with seasonal naive")
    metric_columns = st.columns(3)
    for column, metric in zip(metric_columns, ('mae', 'rmse', 'wape')):
        value = model_metrics[metric]
        naive = naive_metrics[metric]
        value_format = '{:.3f}' if metric == 'wape' else '{:.2f}'
        column.metric(
            metric.upper(),
            value_format.format(value),
            delta=f'{value - naive:+.3f} vs seasonal naive',
            delta_color='inverse',
        )

    review = loaded['frame']
    stores = sorted(review['Store ID'].dropna().unique().tolist())
    if not stores:
        st.info('The aligned prediction frame has no store rows to display.')
        return
    selected_store = st.selectbox('Store ID', stores)
    store_rows = review[review['Store ID'] == selected_store]
    products = sorted(store_rows['Product ID'].dropna().unique().tolist())
    if not products:
        st.info('No products are available for the selected store.')
        return
    selected_product = st.selectbox('Product ID', products)
    selected_rows = store_rows[store_rows['Product ID'] == selected_product].sort_values('Date')
    st.line_chart(selected_rows.set_index('Date')[['actual', 'predicted', 'seasonal_naive']])
    st.dataframe(selected_rows, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner='Loading and engineering features...')
def load_engineered_data():
    config = load_config(str(CONFIG_PATH))
    data_cfg, fe_cfg = config['data'], config['feature_engineering']

    df = load_data(str(DATA_PATH))
    validate_schema(df)
    df = clean_data(df)
    df = df.drop(columns=[c for c in data_cfg['excluded_features'] if c in df.columns])
    df = add_calendar_features(df, DATE_COLUMN)
    df = add_lag_rolling_features(df, [STORE_COLUMN, PRODUCT_COLUMN], TARGET_COLUMN,
                                  fe_cfg['lag_days'], fe_cfg['rolling_windows'])
    lag_rolling_cols = lag_rolling_feature_names(TARGET_COLUMN, fe_cfg['lag_days'], fe_cfg['rolling_windows'])
    df = drop_lag_rolling_warmup(df, lag_rolling_cols)
    return df, config


@st.cache_resource(show_spinner='Loading local model artifact...')
def load_local_model():
    with open(MODEL_DIR / 'model_metadata.json') as f:
        metadata = json.load(f)

    if metadata['model_type'] in ('lstm', 'transformer'):
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_DIR / 'model.keras')
        scaler = joblib.load(MODEL_DIR / 'sequence_scaler.joblib')
        return {'type': 'sequence', 'model': model, 'scaler': scaler, 'metadata': metadata}

    model = joblib.load(MODEL_DIR / 'model.joblib')
    return {'type': 'tabular', 'model': model, 'metadata': metadata}


def build_window(df: pd.DataFrame, store: str, product: str, as_of_date, sequence_length: int,
                 feature_cols: list, overrides: dict):
    """Last `sequence_length` engineered rows for one series, ending at as_of_date."""
    series = df[(df[STORE_COLUMN] == store) & (df[PRODUCT_COLUMN] == product) & (df[DATE_COLUMN] <= as_of_date)]
    series = series.sort_values(DATE_COLUMN).tail(sequence_length)
    if len(series) < sequence_length:
        return None, None

    series = series.copy()
    for col, value in overrides.items():
        if value is not None and col in series.columns:
            series.iloc[-1, series.columns.get_loc(col)] = value

    window = series[feature_cols].to_numpy(dtype=np.float32)[np.newaxis, ...]
    return window, series


def predict_local(bundle: dict, window: np.ndarray) -> float:
    from src.models.sequence_models import predict_sequence_model
    pred = predict_sequence_model(bundle['model'], bundle['scaler'], window)
    return float(clip_non_negative(pred)[0])


def predict_endpoint(endpoint_name: str, region: str, window: np.ndarray) -> dict:
    import boto3
    runtime = boto3.client('runtime.sagemaker', region_name=region)
    payload = json.dumps({'sequence': window[0].tolist()})
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=payload)
    body = json.loads(response['Body'].read().decode('utf-8'))
    return {'payload_sent': payload, 'response': body}


def render_forecast_tester():
    st.subheader('Retail Demand Forecast — Model / Endpoint Tester')
    st.caption('AAI-540 Group 4 — Retail Store Inventory Demand Forecasting')

    df, config = load_engineered_data()
    fe_cfg = config['feature_engineering']
    sequence_length = fe_cfg['sequence_length']

    with st.sidebar:
        st.header('Settings')
        mode = st.radio('Prediction source', ['Local model', 'SageMaker endpoint'])

        endpoint_name, region = None, None
        if mode == 'SageMaker endpoint':
            endpoint_name = st.text_input('Endpoint name', placeholder='retail-forecasting-endpoint-...')
            region = st.text_input('AWS region', value='us-east-1')
            st.caption('Requires AWS credentials available to this process (e.g. `aws configure`, '
                      'or run inside SageMaker Studio with the attached execution role).')

        st.divider()
        store = st.selectbox('Store ID', sorted(df[STORE_COLUMN].unique()))
        product = st.selectbox('Product ID', sorted(df[PRODUCT_COLUMN].unique()))
        as_of_date = st.date_input(
            'Forecast as of (last known day)',
            value=df[DATE_COLUMN].max().date(),
            min_value=df[DATE_COLUMN].min().date(),
            max_value=df[DATE_COLUMN].max().date(),
        )

        st.divider()
        st.caption("What-if overrides for the most recent day (optional)")
        price_override = st.number_input('Price override', min_value=0.0, value=0.0, step=1.0,
                                         help='0 = use the historical value')
        discount_override = st.number_input('Discount % override', min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        promo_override = st.selectbox('Holiday/Promotion override', ['(use historical)', 'No', 'Yes'])

    overrides = {
        'Price': price_override if price_override > 0 else None,
        'Discount': discount_override if discount_override > 0 else None,
        'Holiday/Promotion': {'No': 0, 'Yes': 1}.get(promo_override),
    }

    numeric_cols = (['Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion']
                    + ['day_of_week', 'month', 'quarter', 'year', 'week_of_year', 'is_weekend']
                    + lag_rolling_feature_names(TARGET_COLUMN, fe_cfg['lag_days'], fe_cfg['rolling_windows']))

    as_of_ts = pd.Timestamp(as_of_date)
    window, series = build_window(df, store, product, as_of_ts, sequence_length, numeric_cols, overrides)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f'Recent actual demand — {store} / {product}')
        if series is not None:
            st.line_chart(series.set_index(DATE_COLUMN)[TARGET_COLUMN])
        else:
            st.warning(f'Not enough history before {as_of_date} to build a {sequence_length}-day window.')

    with col2:
        st.subheader('Forecast')
        if window is not None and st.button('Generate forecast', type='primary'):
            if mode == 'Local model':
                bundle = load_local_model()
                if bundle['type'] != 'sequence':
                    st.error('The saved local model is a classical (tabular) model; this tester currently '
                            'supports the sequence (LSTM/Transformer) model only.')
                else:
                    prediction = predict_local(bundle, window)
                    st.metric('Predicted Units Sold (next day)', f'{prediction:.1f}')
                    st.caption(f"Model: {bundle['metadata']['model_type']}")
            else:
                if not endpoint_name:
                    st.error('Enter an endpoint name in the sidebar first.')
                else:
                    try:
                        result = predict_endpoint(endpoint_name, region, window)
                        predictions = result['response'].get('predictions', [])
                        if predictions:
                            st.metric('Predicted Units Sold (next day)', f'{clip_non_negative(np.array(predictions))[0]:.1f}')
                        with st.expander('Raw request / response'):
                            st.json({'request': json.loads(result['payload_sent']), 'response': result['response']})
                    except Exception as exc:
                        st.error(f'Endpoint invocation failed: {exc}')


def main():
    st.set_page_config(page_title='Retail Inventory Forecast Dashboard', layout='wide')
    st.title('Retail Store Inventory Forecast Dashboard')
    st.caption('AAI-540 Group 4 — Retail Store Inventory Demand Forecasting')

    mlflow_config = load_config(str(CONFIG_PATH)).get('mlflow', {})
    with st.sidebar:
        st.header('MLflow settings')
        tracking_uri_input = st.text_input(
            'Tracking URI',
            value=mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db'),
        )
        experiment_name = st.text_input(
            'Experiment name',
            value=mlflow_config.get('experiment_name', 'retail_inventory_forecasting'),
        )
        st.caption(f'Resolved tracking URI: `{resolve_tracking_uri(tracking_uri_input)}`')

    tabs = st.tabs(['Model comparison', 'How predictions are made', 'Forecast tester'])
    with tabs[0]:
        render_model_comparison(resolve_tracking_uri(tracking_uri_input), experiment_name)
    with tabs[1]:
        render_prediction_review()
    with tabs[2]:
        render_forecast_tester()


if __name__ == '__main__':
    main()
