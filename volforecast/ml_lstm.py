# LSTM over the last seq_len days of features, predicting log smoothed realized variance.
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import TensorDataset, DataLoader

from .config import test_start_index
from .features import make_features, make_target


class VolLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)      # predicts log-variance

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(Xs, y, seq_len):
    # Xs (standardized features), y (target) share a DatetimeIndex.
    # Sequence i covers rows [i, i+seq_len) and is labelled with the target at its LAST row,
    # so it never sees a target date's own future.
    # Vectorized with sliding_window_view (bit-identical to the old per-window Python loop).
    Xv = Xs.values.astype(np.float32)
    yv = y.values.astype(np.float32)
    seqs = np.ascontiguousarray(
        sliding_window_view(Xv, seq_len, axis=0).transpose(0, 2, 1))
    targets = yv[seq_len - 1:]
    idx = Xs.index[seq_len - 1:]
    return seqs, targets, idx


def bl_lstm(r, cfg, seq_len=20, hidden=32, layers=1, dropout=0.0,
            epochs=200, patience=15, lr=1e-3, batch_size=64, val_frac=0.15, seed=42):
    torch.manual_seed(seed)

    X = make_features(r)
    y = make_target(r)                        # same smoothed log-variance target as XGBoost
    data = X.join(y.rename("y")).dropna()
    feat_cols = [c for c in data.columns if c != "y"]

    split = data.index[int(len(data) * (1 - cfg["test_frac"]))]
    train_full = data.loc[:split]
    val_i = int(len(train_full) * (1 - val_frac))       # validation = last slice of TRAIN, not test
    train_end = train_full.index[val_i - 1]
    val_end = train_full.index[-1]

    # standardize using TRAIN-only statistics (never fit on val/test)
    mu = train_full.iloc[:val_i][feat_cols].mean()
    sigma = train_full.iloc[:val_i][feat_cols].std().replace(0, 1.0)
    Xs = (data[feat_cols] - mu) / sigma

    seqs, targets, idx = make_sequences(Xs, data["y"], seq_len)
    train_mask = idx <= train_end
    val_mask = (idx > train_end) & (idx <= val_end)
    test_mask = idx > val_end

    Xtr, ytr = torch.tensor(seqs[train_mask]), torch.tensor(targets[train_mask])
    Xval, yval = torch.tensor(seqs[val_mask]), torch.tensor(targets[val_mask])
    Xte = torch.tensor(seqs[test_mask])
    idx_te = idx[test_mask]

    model = VolLSTM(n_features=len(feat_cols), hidden=hidden, layers=layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xval), yval).item()
        if vloss < best_val - 1e-6:
            best_val, bad_epochs = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_var = np.exp(model(Xte).numpy())

    fc = np.full(len(r), np.nan)
    pos = {ts: i for i, ts in enumerate(r.index)}
    for ts, v in zip(idx_te, pred_var):
        fc[pos[ts]] = v
    print(f"LSTM: early-stopped at epoch {epoch + 1}, best val MSE(log-var) = {best_val:.5f}")
    return fc, model
