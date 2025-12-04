


# 4) `data_loader.py` (auto download from kagglehub)

# data_loader.py
"""
Downloads the dataset using kagglehub and provides simple functions
to load a chosen CSV time-series from the downloaded folder.

Assumes the kagglehub library is configured to access Kaggle (if required).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# third-party downloader (per user's request)
import kagglehub

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_benchmark_dataset():
    """
    Downloads the dataset and returns local path where files are extracted.
    """
    print("Downloading dataset with kagglehub...")
    # dataset identifier provided by user
    dataset_id = "giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets"
    # dataset_download returns local path (implementation depends on kagglehub)
    path = kagglehub.dataset_download(dataset_id)
    print("Dataset downloaded to:", path)
    return Path(path)

def list_csv_files(base_path: Path):
    """
    List CSV files inside base_path or its subfolders.
    """
    csvs = list(base_path.rglob("*.csv"))
    return csvs

def load_series_from_csv(csv_path: str, date_col=None, target_col=None, freq=None):
    """
    Load a single time-series CSV into pandas Series or DataFrame.

    Parameters:
    - csv_path: path to the CSV file
    - date_col: name or index of the datetime column (if None, tries to infer)
    - target_col: name of column to forecast (if None and file has >1 column, will take second col)
    - freq: optional frequency string for pd.date_range/fill

    Returns:
    - df: DataFrame with DateTimeIndex and a single numeric target column named 'target'
    """
    df = pd.read_csv(csv_path)
    # If date column not provided, try to find a datetime-like column
    if date_col is None:
        # try common names
        for c in ["date", "timestamp", "ds", "time"]:
            if c in df.columns:
                date_col = c
                break
        # else assume first column is datetime if dtype object
        if date_col is None and df.shape[1] >= 2:
            date_col = df.columns[0]

    if date_col is not None:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
        except Exception:
            # fallback to no datetime index
            pass

    if target_col is None:
        # pick a reasonable target column (not the index)
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                target_col = c
                break

    # create a cleaned DataFrame with a 'target' column
    out = pd.DataFrame({"target": pd.to_numeric(df[target_col], errors="coerce")})
    out = out.dropna()
    if freq:
        out = out.asfreq(freq).interpolate()
    return out

if __name__ == "__main__":
    base = download_benchmark_dataset()
    print("CSV files found:", list_csv_files(base))
