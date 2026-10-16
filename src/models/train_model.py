"""
Classical ML model training module for the tabular (lag/rolling/calendar
feature) view of the forecasting problem.

Every model is a scikit-learn Pipeline (preprocess -> regressor), so the saved
artifact accepts raw feature columns. Hyperparameter search uses
TimeSeriesSplit rather than a shuffled K-fold, since shuffling would leak
future rows into earlier training folds.
"""

from typing import Any, Dict, List

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.data_preprocessing.data_preprocessing import build_preprocessor

MODEL_TYPES = ('ridge', 'random_forest', 'gradient_boosting')


def _make_regressor(model_type: str, random_state: int):
    """Instantiate a regressor with sensible defaults for this dataset."""
    if model_type == 'ridge':
        return Ridge(random_state=random_state)
    if model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                     random_state=random_state)
    if model_type == 'gradient_boosting':
        return HistGradientBoostingRegressor(random_state=random_state)
    raise ValueError(f"Unknown model type: {model_type}")


def build_model(model_type: str, numeric_cols: List[str], categorical_cols: List[str],
                random_state: int = 42) -> Pipeline:
    """
    Build an unfitted preprocess -> regressor pipeline.

    Args:
        model_type: One of MODEL_TYPES
        numeric_cols: Numerical input columns (raw + calendar + lag/rolling)
        categorical_cols: Categorical input columns
        random_state: Random seed

    Returns:
        Unfitted scikit-learn Pipeline
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ('preprocess', build_preprocessor(numeric_cols, categorical_cols)),
        ('regressor', _make_regressor(model_type, random_state)),
    ])


def fit_and_score(model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  numeric_cols: List[str], categorical_cols: List[str],
                  random_state: int = 42) -> Dict[str, Any]:
    """
    Fit a classical model on the training split and score it on validation.

    Args:
        model_type: One of MODEL_TYPES
        X_train, y_train: Training features/target
        X_val, y_val: Validation features/target
        numeric_cols: Numerical input columns
        categorical_cols: Categorical input columns
        random_state: Random seed

    Returns:
        Dictionary with the fitted model and its validation predictions
    """
    model = build_model(model_type, numeric_cols, categorical_cols, random_state)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return {'model_type': model_type, 'model': model, 'val_predictions': val_pred}


# Bounded search spaces (avoid excessively large searches in a course lab environment)
PARAM_DISTRIBUTIONS = {
    'ridge': {
        'regressor__alpha': loguniform(1e-2, 1e2),
    },
    'random_forest': {
        'regressor__n_estimators': randint(100, 400),
        'regressor__max_depth': [8, 12, 16, None],
        'regressor__min_samples_leaf': randint(1, 20),
        'regressor__max_features': ['sqrt', 0.3, 0.5],
    },
    'gradient_boosting': {
        'regressor__learning_rate': loguniform(0.02, 0.3),
        'regressor__max_iter': randint(100, 500),
        'regressor__max_depth': [3, 5, 7, None],
        'regressor__min_samples_leaf': randint(20, 200),
        'regressor__l2_regularization': uniform(0, 1),
    },
}


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, model_type: str,
                          numeric_cols: List[str], categorical_cols: List[str],
                          n_iter: int = 15, n_splits: int = 3,
                          random_state: int = 42) -> Dict[str, Any]:
    """
    Tune hyperparameters with a bounded randomized search over a TimeSeriesSplit.

    Args:
        X_train: Training features, sorted chronologically
        y_train: Training target
        model_type: Type of model to tune (must be in MODEL_TYPES)
        numeric_cols: Numerical input columns
        categorical_cols: Categorical input columns
        n_iter: Number of parameter settings sampled
        n_splits: Number of TimeSeriesSplit folds
        random_state: Random seed

    Returns:
        Dictionary containing the best fitted model, its parameters and CV score
    """
    model = build_model(model_type, numeric_cols, categorical_cols, random_state)
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        model, PARAM_DISTRIBUTIONS[model_type], n_iter=n_iter,
        scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, random_state=random_state,
    )
    search.fit(X_train, y_train)

    return {
        'best_model': search.best_estimator_,
        'best_params': {k.replace('regressor__', ''): v for k, v in search.best_params_.items()},
        'best_score': -search.best_score_,
    }
