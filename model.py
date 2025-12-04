# model.py
"""
Transformer-based time-series forecasting (encoder-decoder style).
This file contains:
- positional encoding
- model: TimeSeriesTransformer
- train and evaluate loops
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ----------------------------
# Positional Encoding (sinusoidal)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd dim, pad last column with zeros
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

# ----------------------------
# TimeSeriesDataset (sliding windows)
# ----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, input_window: int, output_window: int):
        """
        series: 1-D numpy array of target values
        """
        self.series = series.astype(np.float32)
        self.input_window = input_window
        self.output_window = output_window
        self.L = len(series)

    def __len__(self):
        return max(0, self.L - self.input_window - self.output_window + 1)

    def __getitem__(self, idx):
        start = idx
        x = self.series[start : start + self.input_window]
        y = self.series[start + self.input_window : start + self.input_window + self.output_window]
        # reshape to (seq_len, features) -> features=1
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)

# ----------------------------
# Transformer Model
# ----------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        input_window=24,
        output_window=12,
    ):
        super().__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.d_model = d_model

        # input embedding for single feature -> project to d_model
        self.input_proj = nn.Linear(feature_size, d_model)
        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=5000)

        # transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # so input shape is (batch, seq, feature)
        )

        # final projection (d_model -> feature)
        self.output_proj = nn.Linear(d_model, feature_size)

    def forward(self, src, tgt):
        """
        src: (batch, src_seq, feature)
        tgt: (batch, tgt_seq, feature)
        returns: predictions shape (batch, tgt_seq, feature)
        """
        # embed
        src_emb = self.input_proj(src) * math.sqrt(self.d_model)
        tgt_emb = self.input_proj(tgt) * math.sqrt(self.d_model)

        # add position
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # mask for causal decoding (so decoder can't peek future tokens)
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(src.device)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        out = self.output_proj(out)
        return out

    def _generate_square_subsequent_mask(self, sz):
        # PyTorch Transformer expects mask with shape (sz, sz)
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

# ----------------------------
# train / evaluate utilities
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for src, tgt_y in dataloader:
        src = src.to(device)  # (B, in_len, 1)
        tgt_y = tgt_y.to(device)  # (B, out_len, 1)
        # teacher forcing: decoder input is previous true tokens prepended by zeros (or last src value)
        # here we provide a simple start token: last value of src repeated for tgt_len
        decoder_input = src[:, -1:, :].repeat(1, tgt_y.size(1), 1)
        optimizer.zero_grad()
        outputs = model(src, decoder_input)
        loss = criterion(outputs, tgt_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt_y in dataloader:
            src = src.to(device)
            tgt_y = tgt_y.to(device)
            decoder_input = src[:, -1:, :].repeat(1, tgt_y.size(1), 1)
            outputs = model(src, decoder_input)
            loss = criterion(outputs, tgt_y)
            total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)

def predict_forecast(model, series, input_window, output_window, device):
    """
    Autoregressive prediction loop given the last `input_window` values.
    series: numpy array containing the entire series.
    returns: predicted numpy array of length output_window
    """
    model.eval()
    with torch.no_grad():
        src = torch.from_numpy(series[-input_window:].astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
        # initial decoder input: last value repeated
        decoder_input = src[:, -1:, :].repeat(1, output_window, 1)
        preds = model(src, decoder_input)  # (1, output_window, 1)
        return preds.squeeze().cpu().numpy()

# ----------------------------
# convenience: build dataloaders
# ----------------------------
def build_dataloaders(series, input_window, output_window, batch_size=32, val_split=0.1):
    from sklearn.model_selection import train_test_split

    dataset = TimeSeriesDataset(series, input_window, output_window)
    n = len(dataset)
    idx = list(range(n))
    train_idx, val_idx = train_test_split(idx, test_size=val_split, shuffle=False)
    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
