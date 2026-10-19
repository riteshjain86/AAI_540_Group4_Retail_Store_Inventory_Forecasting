"""
Pipeline step 1 (Preprocess): clean the raw retail CSV and split it
chronologically into train / validation / test Parquet files.

Runs inside a SageMaker Processing container, so it mirrors
src/data_ingestion + src/data_preprocessing without importing the project.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

INPUT_DIR = '/opt/ml/processing/input'
OUTPUT_DIR = '/opt/ml/processing'
KEY_COLS = ['Date', 'Store ID', 'Product ID']


def chronological_split(df: pd.DataFrame, val_size: float, test_size: float):
    dates = sorted(df['Date'].unique())
    n_test = max(1, int(round(len(dates) * test_size)))
    n_val = max(1, int(round(len(dates) * val_size)))
    n_train = len(dates) - n_val - n_test
    if n_train <= 0:
        raise ValueError('val_size + test_size leave no dates for training')
    train_cutoff, val_cutoff = dates[n_train - 1], dates[n_train + n_val - 1]
    train = df[df['Date'] <= train_cutoff]
    val = df[(df['Date'] > train_cutoff) & (df['Date'] <= val_cutoff)]
    test = df[df['Date'] > val_cutoff]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    args = parser.parse_args()

    files = sorted(Path(INPUT_DIR).glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files in {INPUT_DIR}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset=KEY_COLS, keep='first')
    df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)

    for name, part in zip(('train', 'validation', 'test'), chronological_split(df, args.val_size, args.test_size)):
        out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        part.reset_index(drop=True).to_parquet(os.path.join(out_dir, f'{name}.parquet'), index=False)
        print(f'{name}: {len(part)} rows, {part["Date"].min().date()} .. {part["Date"].max().date()}')


if __name__ == '__main__':
    main()
