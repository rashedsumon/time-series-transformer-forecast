# utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_test_split_series(series, test_size=0.2):
    n_test = int(len(series) * test_size)
    train = series[:-n_test]
    test = series[-n_test:]
    return train, test

def scale_series(train, test=None, method="minmax"):
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    train_reshaped = train.reshape(-1, 1)
    scaler.fit(train_reshaped)
    train_s = scaler.transform(train_reshaped).flatten()
    if test is not None:
        test_s = scaler.transform(test.reshape(-1, 1)).flatten()
        return scaler, train_s, test_s
    return scaler, train_s

def plot_series(train, test, preds=None, input_window=None):
    plt.figure(figsize=(10, 4))
    n_train = len(train)
    plt.plot(range(n_train), train, label="train")
    plt.plot(range(n_train, n_train + len(test)), test, label="test")
    if preds is not None:
        # preds aligned to start of test
        plt.plot(range(n_train, n_train + len(preds)), preds, label="preds")
    if input_window:
        plt.axvline(n_train - input_window, color="k", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt

def compute_metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
