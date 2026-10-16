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


def main():
    st.set_page_config(page_title='Retail Demand Forecast Tester', layout='wide')
    st.title('Retail Demand Forecast — Model / Endpoint Tester')
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


if __name__ == '__main__':
    main()
