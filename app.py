# app.py
"""
Streamlit app to demonstrate training and forecasting with a Transformer for time-series.
Main features:
- download dataset via kagglehub (data_loader)
- pick a CSV series to use
- configure windows and model hyperparameters
- train model (quick small epochs for demo)
- visualize forecast
"""

import streamlit as st
from pathlib import Path
import numpy as np
import torch
import time

# local modules
from data_loader import download_benchmark_dataset, list_csv_files, load_series_from_csv
from model import TimeSeriesTransformer, build_dataloaders, train_epoch, evaluate, predict_forecast
from utils import train_test_split_series, scale_series, plot_series, compute_metrics

st.set_page_config(page_title="Time-Series Transformer Demo", layout="wide")

st.title("Time-Series Transformer Forecasting — Demo")

# Sidebar: dataset download and selection
st.sidebar.header("Dataset")
if st.sidebar.button("Download dataset from KaggleHub"):
    with st.spinner("Downloading dataset..."):
        base = download_benchmark_dataset()
        st.success(f"Downloaded to: {base}")
else:
    base = Path("data")
    if not base.exists():
        st.info("No dataset found in `data/`. Click 'Download dataset' on the sidebar to fetch it.")

# list CSVs after download (non-blocking)
if base.exists():
    csvs = list_csv_files(base)
    csv_options = [str(p) for p in csvs]
    chosen = st.sidebar.selectbox("Choose CSV file", options=csv_options)
else:
    csv_options = []
    chosen = None

# Model hyperparameters & windows
st.sidebar.header("Model & Window")
input_window = st.sidebar.number_input("Input window (timesteps)", value=48, min_value=4, max_value=1024, step=1)
output_window = st.sidebar.number_input("Forecast horizon (timesteps)", value=12, min_value=1, max_value=256, step=1)
d_model = st.sidebar.selectbox("d_model", options=[32, 64, 128], index=1)
nhead = st.sidebar.selectbox("nhead", options=[2, 4, 8], index=1)
batch_size = st.sidebar.selectbox("batch_size", options=[8, 16, 32, 64], index=2)
epochs = st.sidebar.number_input("Epochs", value=5, min_value=1, max_value=100, step=1)
lr = st.sidebar.number_input("Learning rate", value=1e-3, format="%.6f")

# main area
if chosen:
    st.write("Selected file:", chosen)
    st.info("Loading series...")
    df = load_series_from_csv(chosen)
    # take first (or only) column 'target'
    series = df["target"].values
    st.write(f"Series length: {len(series)}")
    st.line_chart(series)

    # train/test split and scaling
    test_fraction = st.sidebar.slider("Test fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    train_s, test_s = train_test_split_series(series, test_size=test_fraction)
    scaler, train_scaled, test_scaled = scale_series(train_s, test_s, method="minmax")

    # button to start quick training
    if st.button("Train model (demo)"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write("Using device:", device)
        # construct dataloaders from scaled train only (we'll forecast on test)
        # combine train + test for sliding windows input (model builder expects whole series optionally)
        full_scaled = np.concatenate([train_scaled, test_scaled])
        train_loader, val_loader = build_dataloaders(full_scaled, input_window, output_window, batch_size=batch_size, val_split=0.15)

        model = TimeSeriesTransformer(
            feature_size=1,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=4 * d_model,
            input_window=input_window,
            output_window=output_window,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        progress_text = st.empty()
        pbar = st.progress(0)
        best_val = float("inf")
        for epoch in range(1, int(epochs) + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0
            progress_text.text(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f} — val_loss: {val_loss:.6f} — {elapsed:.1f}s")
            pbar.progress(int(epoch / epochs * 100))
            if val_loss < best_val:
                best_val = val_loss
                # optional: save best model
                torch.save(model.state_dict(), "models/best_model.pth")
        st.success("Training finished!")

        # Predict on test set: perform rolling forecasts across test segment
        preds = []
        # create a combined scaled array (train + test)
        combined = full_scaled.copy()
        # we will forecast the test segment step-by-step (non-overlapping) OR forecast entire horizon from last train window
        # For simplicity show multi-step forecast from the last input window of the training set
        last_train_window = combined[: len(train_scaled)][-input_window:]
        # model expects numpy sequence; call predict_forecast to get output_window predictions
        forecast_scaled = predict_forecast(model, np.concatenate([combined[: len(train_scaled)], combined[len(train_scaled):]]), input_window, output_window, device)
        # inverse scale
        # forecast was in scaled space -> inverse
        forecast_unscaled = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        # Build display arrays
        train_unscaled = train_s
        test_unscaled = test_s
        st.write("Forecast (first horizon):")
        st.line_chart(np.concatenate([train_unscaled, test_unscaled]))
        st.write("Overlayed forecast (first horizon):")
        # plot using matplotlib for overlay
        plt = plot_series(train_unscaled, test_unscaled, preds=forecast_unscaled, input_window=input_window)
        st.pyplot(plt)

        metrics = compute_metrics(test_unscaled[: len(forecast_unscaled)], forecast_unscaled[: len(test_unscaled)])
        st.write("Metrics (on forecast vs test slice):", metrics)

    st.write("---")
    st.write("Tip: this demo trains a small model for learning purposes. For production:")
    st.write("- increase data, tune hyperparameters, consider learning rate schedulers, early stopping.")
    st.write("- consider Multivariate inputs by expanding `feature_size` and feeding multiple features.")
else:
    st.info("No CSV selected yet. Please download dataset and select a CSV from the sidebar.")
