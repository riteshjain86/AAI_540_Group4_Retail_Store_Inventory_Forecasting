"""
Trivial seasonal-naive baseline for demand forecasting.

A credible forecasting project must show that any learned model (classical ML
or deep learning) actually beats a baseline this simple, since weekly
seasonality alone explains a large share of retail demand patterns.
"""

import numpy as np
import pandas as pd


class SeasonalNaiveModel:
    """Predicts today's demand as the actual demand observed `lag` days ago."""

    def __init__(self, target_col: str, lag: int = 7):
        self.target_col = target_col
        self.lag = lag
        self.lag_column = f'{target_col}_lag_{lag}'

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None):
        """No-op: the seasonal-naive baseline has no parameters to learn."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Args:
            X: DataFrame containing the precomputed f'{target_col}_lag_{lag}' column

        Returns:
            Array of predictions
        """
        if self.lag_column not in X.columns:
            raise ValueError(f"Expected lag feature '{self.lag_column}' in input columns")
        return X[self.lag_column].to_numpy()
