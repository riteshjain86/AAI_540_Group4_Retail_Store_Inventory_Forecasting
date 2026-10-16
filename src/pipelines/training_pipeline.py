"""
Training pipeline for end-to-end demand forecasting.

Steps: load -> validate -> clean -> calendar/lag/rolling feature engineering ->
chronological 70/15/15 split -> seasonal-naive baseline -> classical ML
(Ridge / Random Forest / Gradient Boosting) with time-series-aware tuning ->
LSTM and Transformer sequence models -> pick the best candidate on validation
WAPE -> evaluate once on test -> quality gates -> log to MLflow and save the
selected model for serving.
"""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.data_ingestion.data_ingestion import (
    load_data, validate_schema, chronological_cutoffs,
    DATE_COLUMN, STORE_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN,
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
)
from src.data_preprocessing.data_preprocessing import clean_data, clip_non_negative
from src.feature_engineering.feature_engineering import (
    add_calendar_features, add_lag_rolling_features, drop_lag_rolling_warmup,
    build_sequences, lag_rolling_feature_names, CALENDAR_FEATURES,
)
from src.models.baseline import SeasonalNaiveModel
from src.models.train_model import MODEL_TYPES as CLASSICAL_MODEL_TYPES, fit_and_score, hyperparameter_tuning
from src.models.sequence_models import SEQUENCE_MODEL_BUILDERS, train_sequence_model, predict_sequence_model
from src.models.predict_model import evaluate_forecast
from src.utils.helpers import load_config, ensure_dir

GROUP_COLS = [STORE_COLUMN, PRODUCT_COLUMN]
# Features that are only known after the fact or are themselves a forecast of
# the target (see config.yaml data.excluded_features) are dropped upstream.
RAW_NUMERIC_FEATURES = ['Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion']


class TrainingPipeline:
    """End-to-end training pipeline for retail demand forecasting."""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)

    def run(self, data_path: str):
        data_cfg = self.config['data']
        train_cfg = self.config['training']
        fe_cfg = self.config['feature_engineering']
        random_state = train_cfg.get('random_state', 42)
        candidates = train_cfg['candidate_models']

        print("Loading data...")
        df = load_data(data_path)
        quality_report = validate_schema(df)
        print(f"Data quality: {quality_report}")

        df = clean_data(df)
        df = df.drop(columns=[c for c in data_cfg.get('excluded_features', []) if c in df.columns])

        if fe_cfg.get('add_calendar_features', True):
            df = add_calendar_features(df, DATE_COLUMN)
        df = add_lag_rolling_features(df, GROUP_COLS, TARGET_COLUMN,
                                      fe_cfg['lag_days'], fe_cfg['rolling_windows'])

        lag_rolling_cols = lag_rolling_feature_names(TARGET_COLUMN, fe_cfg['lag_days'], fe_cfg['rolling_windows'])
        df = drop_lag_rolling_warmup(df, lag_rolling_cols)

        numeric_cols = [c for c in RAW_NUMERIC_FEATURES if c in df.columns] + CALENDAR_FEATURES + lag_rolling_cols
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns] + GROUP_COLS

        train_cutoff, val_cutoff = chronological_cutoffs(df[DATE_COLUMN], data_cfg['val_size'], data_cfg['test_size'])
        train_df = df[df[DATE_COLUMN] <= train_cutoff].reset_index(drop=True)
        val_df = df[(df[DATE_COLUMN] > train_cutoff) & (df[DATE_COLUMN] <= val_cutoff)].reset_index(drop=True)
        test_df = df[df[DATE_COLUMN] > val_cutoff].reset_index(drop=True)
        print(f"Split sizes - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} "
              f"(cutoffs: {train_cutoff.date()} / {val_cutoff.date()})")
        self._save_splits(train_df, val_df, test_df)

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name='training_pipeline') as run:
            mlflow.log_params({
                'excluded_features': ','.join(data_cfg.get('excluded_features', [])),
                'sequence_length': fe_cfg['sequence_length'],
                'n_train': len(train_df), 'n_val': len(val_df), 'n_test': len(test_df),
            })
            mlflow.log_dict(quality_report, 'data_quality_report.json')

            results = {}

            # --- Seasonal-naive baseline ---
            if 'seasonal_naive' in candidates:
                results['seasonal_naive'] = self._run_naive(val_df, train_cfg['selection_metric'])

            # --- Classical ML ---
            classical_candidates = [m for m in candidates if m in CLASSICAL_MODEL_TYPES]
            if classical_candidates:
                best_classical = self._run_classical(
                    train_df, val_df, classical_candidates, numeric_cols, categorical_cols, random_state,
                )
                results['classical'] = best_classical

            # --- Deep learning (LSTM / Transformer) ---
            dl_candidates = [m for m in candidates if m in SEQUENCE_MODEL_BUILDERS]
            if dl_candidates:
                dl_feature_cols = numeric_cols
                X_all, y_all, meta_all = build_sequences(
                    df, dl_feature_cols, TARGET_COLUMN, GROUP_COLS, DATE_COLUMN, fe_cfg['sequence_length'],
                )
                seq_train_mask = meta_all[DATE_COLUMN] <= train_cutoff
                seq_val_mask = (meta_all[DATE_COLUMN] > train_cutoff) & (meta_all[DATE_COLUMN] <= val_cutoff)
                seq_test_mask = meta_all[DATE_COLUMN] > val_cutoff
                print(f"Sequence samples - train: {seq_train_mask.sum()}, val: {seq_val_mask.sum()}, "
                      f"test: {seq_test_mask.sum()}")

                for model_type in dl_candidates:
                    result = self._run_sequence_model(
                        model_type, X_all, y_all, seq_train_mask, seq_val_mask,
                        train_cfg['deep_learning'], train_cfg['selection_metric'], random_state,
                    )
                    results[model_type] = result

            # --- Pick the overall best candidate on validation WAPE ---
            comparison = pd.DataFrame([
                {'model_type': name, **r['val_metrics']} for name, r in results.items()
            ]).sort_values(train_cfg['selection_metric']).reset_index(drop=True)
            mlflow.log_table(comparison, 'model_comparison.json')
            print("\n=== Validation comparison ===")
            print(comparison.to_string(index=False))

            best_name = comparison.iloc[0]['model_type']
            best = results[best_name]
            print(f"\nSelected model: {best_name}")

            # --- Final, single evaluation on the untouched test split ---
            test_metrics, test_predictions = self._evaluate_on_test(
                best_name, best, test_df, X_all if dl_candidates else None,
                y_all if dl_candidates else None, seq_test_mask if dl_candidates else None,
            )
            naive_test_metrics = self._evaluate_naive_on_test(test_df)

            mlflow.log_params({'selected_model': best_name, **best.get('params', {})})
            mlflow.log_metrics({f'val_{k}': v for k, v in best['val_metrics'].items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            mlflow.log_metrics({f'test_naive_{k}': v for k, v in naive_test_metrics.items()})

            mae_improvement = (naive_test_metrics['mae'] - test_metrics['mae']) / naive_test_metrics['mae']
            mlflow.log_metric('mae_improvement_over_naive', mae_improvement)

            gates = train_cfg.get('quality_gates', {})
            failed_gates = {}
            if 'wape' in gates and test_metrics['wape'] > gates['wape']:
                failed_gates['wape'] = test_metrics['wape']
            if 'mae_improvement_over_naive' in gates and mae_improvement < gates['mae_improvement_over_naive']:
                failed_gates['mae_improvement_over_naive'] = mae_improvement
            mlflow.set_tag('quality_gates_passed', str(not failed_gates))

            self._print_report(best_name, test_metrics, naive_test_metrics, mae_improvement)
            if failed_gates:
                raise RuntimeError(f"Model failed quality gates {gates}: {failed_gates}")

            metadata = {
                'run_id': run.info.run_id,
                'model_type': best_name,
                'params': best.get('params', {}),
                'numeric_cols': numeric_cols,
                'categorical_cols': categorical_cols,
                'sequence_length': fe_cfg['sequence_length'],
                'excluded_features': data_cfg.get('excluded_features', []),
                'test_metrics': test_metrics,
            }
            self._save_model(best_name, best, metadata)

        return {'best_model': best_name, 'comparison': comparison, 'test_metrics': test_metrics,
               'run_id': run.info.run_id}

    def _run_naive(self, val_df: pd.DataFrame, selection_metric: str):
        model = SeasonalNaiveModel(TARGET_COLUMN, lag=7)
        val_pred = clip_non_negative(model.predict(val_df))
        metrics = evaluate_forecast(val_df[TARGET_COLUMN], val_pred)
        with mlflow.start_run(run_name='seasonal_naive', nested=True):
            mlflow.log_metrics({f'val_{k}': v for k, v in metrics.items()})
        print(f"  {'seasonal_naive':<20} WAPE={metrics['wape']:.4f}  MAE={metrics['mae']:.2f}")
        return {'model': model, 'val_metrics': metrics, 'params': {}}

    def _run_classical(self, train_df, val_df, candidates, numeric_cols, categorical_cols, random_state):
        X_train, y_train = train_df[numeric_cols + categorical_cols], train_df[TARGET_COLUMN]
        X_val, y_val = val_df[numeric_cols + categorical_cols], val_df[TARGET_COLUMN]

        scored = []
        for model_type in candidates:
            result = fit_and_score(model_type, X_train, y_train, X_val, y_val,
                                   numeric_cols, categorical_cols, random_state)
            val_pred = clip_non_negative(result['val_predictions'])
            metrics = evaluate_forecast(y_val, val_pred)
            scored.append({'model_type': model_type, 'metrics': metrics})
            with mlflow.start_run(run_name=f'classical__{model_type}', nested=True):
                mlflow.log_params({'model_type': model_type})
                mlflow.log_metrics({f'val_{k}': v for k, v in metrics.items()})
            print(f"  {model_type:<20} WAPE={metrics['wape']:.4f}  MAE={metrics['mae']:.2f}")

        best = min(scored, key=lambda r: r['metrics']['wape'])
        print(f"  Tuning best classical candidate: {best['model_type']}")
        tuned = hyperparameter_tuning(X_train, y_train, best['model_type'], numeric_cols, categorical_cols,
                                      n_iter=15, random_state=random_state)
        tuned_pred = clip_non_negative(tuned['best_model'].predict(X_val))
        tuned_metrics = evaluate_forecast(y_val, tuned_pred)
        with mlflow.start_run(run_name=f"classical__{best['model_type']}__tuned", nested=True):
            mlflow.log_params(tuned['best_params'])
            mlflow.log_metrics({f'val_{k}': v for k, v in tuned_metrics.items()})
        print(f"  {'tuned ' + best['model_type']:<20} WAPE={tuned_metrics['wape']:.4f}  MAE={tuned_metrics['mae']:.2f}")

        return {'model': tuned['best_model'], 'val_metrics': tuned_metrics,
               'params': {'model_type': best['model_type'], **tuned['best_params']}}

    def _run_sequence_model(self, model_type, X_all, y_all, train_mask, val_mask, dl_config,
                            selection_metric, random_state):
        X_train, y_train = X_all[train_mask.to_numpy()], y_all[train_mask.to_numpy()]
        X_val, y_val = X_all[val_mask.to_numpy()], y_all[val_mask.to_numpy()]

        model, extras = train_sequence_model(model_type, X_train, y_train, X_val, y_val, dl_config, random_state)
        val_pred = clip_non_negative(predict_sequence_model(model, extras['scaler'], X_val))
        metrics = evaluate_forecast(y_val, val_pred)

        with mlflow.start_run(run_name=model_type, nested=True):
            mlflow.log_params({'model_type': model_type, **dl_config.get(model_type, {}),
                               'epochs_run': len(extras['history']['loss'])})
            mlflow.log_metrics({f'val_{k}': v for k, v in metrics.items()})
            for epoch, (loss, val_loss) in enumerate(zip(extras['history']['loss'], extras['history']['val_loss'])):
                mlflow.log_metrics({'train_loss': loss, 'val_loss': val_loss}, step=epoch)
        print(f"  {model_type:<20} WAPE={metrics['wape']:.4f}  MAE={metrics['mae']:.2f}")

        return {'model': model, 'scaler': extras['scaler'], 'val_metrics': metrics,
               'params': {'model_type': model_type}}

    def _evaluate_on_test(self, best_name, best, test_df, X_all, y_all, seq_test_mask):
        if best_name == 'seasonal_naive':
            pred = clip_non_negative(best['model'].predict(test_df))
            return evaluate_forecast(test_df[TARGET_COLUMN], pred), pred
        if best_name in SEQUENCE_MODEL_BUILDERS:
            X_test = X_all[seq_test_mask.to_numpy()]
            y_test = y_all[seq_test_mask.to_numpy()]
            pred = clip_non_negative(predict_sequence_model(best['model'], best['scaler'], X_test))
            return evaluate_forecast(y_test, pred), pred
        # classical model: features were a subset of test_df's columns
        feature_cols = best['model'].named_steps['preprocess'].feature_names_in_
        pred = clip_non_negative(best['model'].predict(test_df[feature_cols]))
        return evaluate_forecast(test_df[TARGET_COLUMN], pred), pred

    @staticmethod
    def _evaluate_naive_on_test(test_df):
        model = SeasonalNaiveModel(TARGET_COLUMN, lag=7)
        pred = clip_non_negative(model.predict(test_df))
        return evaluate_forecast(test_df[TARGET_COLUMN], pred)

    def _save_splits(self, train_df, val_df, test_df):
        """Persist the splits as Parquet so notebooks and later stages reuse identical data."""
        out_dir = Path(self.config['data']['processed_path'])
        ensure_dir(out_dir)
        for name, part in [('train', train_df), ('val', val_df), ('test', test_df)]:
            part.to_parquet(out_dir / f'{name}.parquet', index=False)

    def _save_model(self, model_type, best, metadata):
        """Save the selected model (classical sklearn Pipeline or Keras sequence model) for serving."""
        out_dir = Path(self.config['model']['output_dir'])
        ensure_dir(out_dir)

        if model_type == 'seasonal_naive':
            joblib.dump(best['model'], out_dir / 'model.joblib')
        elif model_type in SEQUENCE_MODEL_BUILDERS:
            best['model'].save(out_dir / 'model.keras')
            joblib.dump(best['scaler'], out_dir / 'sequence_scaler.joblib')
        else:
            joblib.dump(best['model'], out_dir / 'model.joblib')

        with open(out_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Model saved to {out_dir}")

    @staticmethod
    def _print_report(best_name, test_metrics, naive_test_metrics, mae_improvement):
        print("\n=== Test set results ===")
        print(f"Selected model: {best_name}")
        print(f"MAE={test_metrics['mae']:.2f}  RMSE={test_metrics['rmse']:.2f}  "
              f"WAPE={test_metrics['wape']:.4f}  sMAPE={test_metrics['smape']:.4f}")
        print(f"Seasonal-naive baseline: MAE={naive_test_metrics['mae']:.2f}  WAPE={naive_test_metrics['wape']:.4f}")
        print(f"MAE improvement over naive: {mae_improvement:.1%}")


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    data_path = pipeline.config.get('data', {}).get('raw_path', 'data/raw/retail_store_inventory.csv')
    pipeline.run(data_path=data_path)
