"""
LSTM + GAT training skeleton for stock 5-day excess return prediction.

Assumptions for input data (a single CSV/Parquet):
- columns: date, stock, feature_1, feature_2, ..., future_ret_5d, market_ret_5d
- future_ret_5d = stock raw return over next 5 business days
- market_ret_5d  = benchmark return over next 5 business days
We compute target = future_ret_5d - market_ret_5d (excess return).
Adjust paths / field names as needed.

Requirements:
  pip install torch torchvision torchaudio
  pip install torch-geometric (follow instructions for your CUDA)
  pip install pandas numpy scikit-learn statsmodels tqdm

Usage:
  - Configure DATA_PATH, feature columns, and hyperparams below.
  - Run: python train_lstm_gat.py
"""
import os
import math
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# PyG
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf

# --------------------------
# Config / Hyperparameters
# --------------------------
DATA_PATH = "data/historical.csv"   # change to your data file
DATE_COL = "date"
STOCK_COL = "stock"
FUTURE_RET_COL = "future_ret_5d"
MARKET_RET_COL = "market_ret_5d"

FEATURE_COLS = None  # if None infer from CSV as all columns between stock and FUTURE_RET_COL
TIME_WINDOW = 60      # days lookback for sequence features
PRED_HORIZON = 5

TOPK = 10             # top-k edges per node from correlation matrix
BATCH_STOCK_SUBSAMPLE = None  # if int, sample that many stocks per batch to reduce memory

HIDDEN_LSTM = 128
GAT_HIDDEN = 64
GAT_HEADS = 4
MLP_HIDDEN = 64

LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_IC = 1.0  # weight for negative-pearson(IC) loss
TOPK_PORTFOLIO = 30

# --------------------------
# Utilities: statistics
# --------------------------
def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

def newey_west_t(returns: np.ndarray, nlags: int = 5) -> Tuple[float, float]:
    """
    Compute Newey-West adjusted t-stat for mean(returns).
    returns: daily P&L series (excess returns).
    nlags: number of lags for NW estimator.
    Returns: (t_stat, se)
    """
    returns = np.asarray(returns)
    n = len(returns)
    mean = returns.mean()
    # autocovariances (biased) via acovf
    gamma = acovf(returns, fft=True, demean=True)[: nlags + 1]  # gamma_0 ... gamma_L
    # Bartlett weights
    s = gamma[0] + 2.0 * sum((1.0 - (l / (nlags + 1.0))) * gamma[l] for l in range(1, nlags + 1))
    var_mean = s / n
    se = math.sqrt(var_mean) if var_mean > 0 else 1e-8
    t = mean / se
    return t, se

# --------------------------
# Dataset preparation
# --------------------------
class PanelDataset(Dataset):
    """
    Build time-series windows for all stocks for a given target date.
    Each item is:
      - X: tensor shape (num_stocks, time_window, num_features)
      - y: array shape (num_stocks,) of target excess return
      - stock_list: list of stock ids in order
      - date: target date (the day we predict for)
    Note: This simple implementation materializes per date. Adapt to streaming as needed.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 time_window: int = 60, min_stocks: int = 10):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.time_window = time_window

        # pivot by stock/date for fast slicing
        self.dates = sorted(self.df[DATE_COL].unique())
        self.stocks = sorted(self.df[STOCK_COL].unique())
        # build a dict mapping (stock -> series of features indexed by date)
        self.panel = {}
        for s in self.stocks:
            tmp = self.df[self.df[STOCK_COL] == s].set_index(DATE_COL).sort_index()
            self.panel[s] = tmp

        # construct valid target dates (where for all stocks we have at least time_window history and target)
        self.valid_dates = []
        for d in self.dates:
            # check if at least min_stocks have data for window and target
            count = 0
            for s in self.stocks:
                ts = self.panel[s]
                if d in ts.index:
                    idx = ts.index.get_loc(d)
                    if idx - (time_window - 1) >= 0:
                        # ensure target exists at date d
                        if FUTURE_RET_COL in ts.columns and MARKET_RET_COL in ts.columns:
                            count += 1
            if count >= min_stocks:
                self.valid_dates.append(d)

    def __len__(self):
        return len(self.valid_dates)

    def __getitem__(self, idx):
        d = self.valid_dates[idx]
        rows = []
        targets = []
        stock_list = []
        for s in self.stocks:
            ts = self.panel[s]
            if d in ts.index:
                i = ts.index.get_loc(d)
                if i - (self.time_window - 1) >= 0:
                    window_df = ts.iloc[i - (self.time_window - 1) : i + 1]  # inclusive, length time_window
                    if set(self.feature_cols).issubset(window_df.columns) and FUTURE_RET_COL in ts.columns and MARKET_RET_COL in ts.columns:
                        feat = window_df[self.feature_cols].values  # (time_window, nfeat)
                        future = ts.iloc[i][FUTURE_RET_COL]
                        mkt = ts.iloc[i][MARKET_RET_COL]
                        target = future - mkt
                        rows.append(feat)
                        targets.append(target)
                        stock_list.append(s)
        X = np.stack(rows, axis=0)  # (num_stocks, time_window, nfeat)
        y = np.array(targets, dtype=np.float32)
        return {
            "date": d,
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "stocks": stock_list
        }

# --------------------------
# Graph construction utilities
# --------------------------
def build_correlation_graph(returns_matrix: np.ndarray, topk: int = 10) -> torch.LongTensor:
    """
    returns_matrix: (num_stocks, window_len) daily returns used to compute correlations (or residuals)
    Return edge_index shape [2, E] for PyG
    Approach: compute pairwise Pearson corr matrix, for each node pick topk abs(corr) neighbors (exclude self)
    """
    num = returns_matrix.shape[0]
    if num <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    corr = np.corrcoef(returns_matrix)  # (num, num)
    corr[np.isnan(corr)] = 0.0
    edges = set()
    for i in range(num):
        row = corr[i].copy()
        row[i] = 0.0
        idx = np.argsort(-np.abs(row))[:topk]
        for j in idx:
            if i == j:
                continue
            edges.add((i, j))
            edges.add((j, i))
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edges = np.array(list(edges)).T  # shape (2, E)
    return torch.tensor(edges, dtype=torch.long)

# --------------------------
# Model
# --------------------------
class LSTMGATModel(nn.Module):
    def __init__(self, in_feat: int, lstm_hidden: int, gat_hidden: int,
                 gat_heads: int, mlp_hidden: int):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=lstm_hidden, batch_first=True)
        # GAT expects node features shape (num_nodes, lstm_hidden)
        self.gat1 = GATConv(lstm_hidden, gat_hidden, heads=gat_heads, concat=True, dropout=0.1)
        self.gat2 = GATConv(gat_hidden * gat_heads, lstm_hidden, heads=1, concat=False, dropout=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, X: torch.Tensor, edge_index: torch.LongTensor):
        # X: (num_nodes, time_window, nfeat)
        num_nodes = X.shape[0]
        # pass each node's time series through shared LSTM -> get last hidden
        out, (h_n, c_n) = self.lstm(X)  # out: (num_nodes, time_window, lstm_hidden)
        h_last = out[:, -1, :]  # (num_nodes, lstm_hidden)

        if edge_index.numel() == 0:
            h_g = h_last
        else:
            # PyG GAT expects x shape [N, F] and edge_index [2, E]
            x = h_last
            x = self.gat1(x, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            h_g = x  # (num_nodes, lstm_hidden)
        preds = self.mlp(h_g).squeeze(-1)  # (num_nodes,)
        return preds, h_g

# --------------------------
# Losses: MSE + negative Pearson(IC)
# --------------------------
def negative_pearson_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # compute over batch of nodes
    preds_center = preds - preds.mean()
    targets_center = targets - targets.mean()
    num = preds_center.shape[0]
    denom = torch.sqrt((preds_center ** 2).sum() * (targets_center ** 2).sum()) + 1e-8
    corr = (preds_center * targets_center).sum() / denom
    # negative because we want to maximize corr
    return -corr

# --------------------------
# Evaluation helpers
# --------------------------
def compute_ic(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    pear = pearson_corr(preds, targets)
    # spearman
    try:
        from scipy.stats import spearmanr
        spear = spearmanr(preds, targets).correlation
    except Exception:
        spear = 0.0
    return {"pearson": float(pear), "spearman": float(spear)}

def topk_backtest(preds: np.ndarray, targets: np.ndarray, topk: int = 30) -> Dict:
    # long-only topk equally-weighted portfolio returns (single-period)
    idx = np.argsort(-preds)[:topk]
    port_ret = targets[idx].mean()
    return {"topk_mean_excess_return": float(port_ret)}

# --------------------------
# Walk-forward training loop
# --------------------------
def walk_forward_splits(dates: List[str], train_window: int, val_window: int, test_window: int, step: int = None):
    """
    Yield (train_dates, val_dates, test_dates) windows for walk-forward.
    train_window, val_window, test_window: number of days
    step: sliding step; default = test_window
    """
    if step is None:
        step = test_window
    D = len(dates)
    i = 0
    while i + train_window + val_window + test_window <= D:
        train = dates[i : i + train_window]
        val = dates[i + train_window : i + train_window + val_window]
        test = dates[i + train_window + val_window : i + train_window + val_window + test_window]
        yield train, val, test
        i += step

def train_and_evaluate(df: pd.DataFrame, feature_cols: List[str]):
    # instantiate dataset once; we will filter via dates
    ds = PanelDataset(df, feature_cols, time_window=TIME_WINDOW)
    # Build date list
    all_dates = ds.valid_dates
    # walk-forward config (example)
    train_window = 500
    val_window = 60
    test_window = 60
    results = []
    for train_dates, val_dates, test_dates in walk_forward_splits(all_dates, train_window, val_window, test_window):
        print(f"WF split: train {train_dates[0]}..{train_dates[-1]} | val {val_dates[0]}..{val_dates[-1]} | test {test_dates[0]}..{test_dates[-1]}")
        # slice df for train/val/test by date
        df_train = df[df[DATE_COL].isin(train_dates)].copy()
        df_val = df[df[DATE_COL].isin(val_dates)].copy()
        df_test = df[df[DATE_COL].isin(test_dates)].copy()

        model = LSTMGATModel(in_feat=len(feature_cols), lstm_hidden=HIDDEN_LSTM,
                             gat_hidden=GAT_HIDDEN, gat_heads=GAT_HEADS, mlp_hidden=MLP_HIDDEN).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val_ic = -999
        best_state = None

        # create datasets for training per-day to allow dynamic graph
        train_ds = PanelDataset(df_train, feature_cols, time_window=TIME_WINDOW)
        val_ds = PanelDataset(df_val, feature_cols, time_window=TIME_WINDOW)
        test_ds = PanelDataset(df_test, feature_cols, time_window=TIME_WINDOW)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            # iterate per-date sample (each sample is multi-stock panel)
            for i in range(len(train_ds)):
                sample = train_ds[i]
                X = sample["X"].to(DEVICE)  # (num_nodes, time_window, nfeat)
                y = sample["y"].to(DEVICE)
                num_nodes = X.shape[0]
                if num_nodes == 0:
                    continue
                # build dynamic returns window for corr graph using price returns from X's last feature if provided;
                # here we assume return feature is included as one of feature_cols named 'ret_1d' etc.
                # For demo: compute corr across the last-column of X (if present)
                # Fallback: random sparse graph if insufficient info
                try:
                    # pick column 0 as proxy returns if no explicit returns
                    returns_window = X[:, :, 0].cpu().numpy()
                    edge_index = build_correlation_graph(returns_window, topk=TOPK).to(DEVICE)
                except Exception:
                    edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE)

                opt.zero_grad()
                preds, _ = model(X, edge_index)
                mse = nn.functional.mse_loss(preds, y)
                neg_ic = negative_pearson_loss(preds, y)
                loss = mse + LAMBDA_IC * neg_ic
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
            # validation
            model.eval()
            val_ics = []
            val_topk_returns = []
            with torch.no_grad():
                for i in range(len(val_ds)):
                    sample = val_ds[i]
                    X = sample["X"].to(DEVICE)
                    y = sample["y"].cpu().numpy()
                    if X.shape[0] == 0:
                        continue
                    try:
                        returns_window = X[:, :, 0].cpu().numpy()
                        edge_index = build_correlation_graph(returns_window, topk=TOPK).to(DEVICE)
                    except Exception:
                        edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
                    preds, _ = model(X, edge_index)
                    preds_np = preds.detach().cpu().numpy()
                    ic = pearson_corr(preds_np, y)
                    val_ics.append(ic)
                    bk = topk_backtest(preds_np, y, topk=min(TOPK_PORTFOLIO, len(y)))
                    val_topk_returns.append(bk["topk_mean_excess_return"])
            mean_val_ic = np.nanmean(val_ics) if len(val_ics) > 0 else 0.0
            mean_val_ret = np.nanmean(val_topk_returns) if len(val_topk_returns) > 0 else 0.0
            print(f"Epoch {epoch} loss {total_loss:.4f} val_ic {mean_val_ic:.5f} val_topk_ret {mean_val_ret:.5f}")
            if mean_val_ic > best_val_ic:
                best_val_ic = mean_val_ic
                best_state = model.state_dict()

        # after training evaluate on test window using best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        test_preds = []
        test_targets = []
        dates_for_test = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                sample = test_ds[i]
                X = sample["X"].to(DEVICE)
                y = sample["y"].cpu().numpy()
                if X.shape[0] == 0:
                    continue
                try:
                    returns_window = X[:, :, 0].cpu().numpy()
                    edge_index = build_correlation_graph(returns_window, topk=TOPK).to(DEVICE)
                except Exception:
                    edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
                preds, _ = model(X, edge_index)
                preds_np = preds.detach().cpu().numpy()
                test_preds.append(preds_np)
                test_targets.append(y)
                dates_for_test.append(sample["date"])
        # flatten across dates -> compute daily portfolio P&L (topk)
        daily_pnls = []
        for p, t in zip(test_preds, test_targets):
            bk = topk_backtest(p, t, topk=min(TOPK_PORTFOLIO, len(p)))
            daily_pnls.append(bk["topk_mean_excess_return"])
        daily_pnls = np.array(daily_pnls)
        mean_ic = np.nanmean([pearson_corr(p, t) for p, t in zip(test_preds, test_targets)])
        nw_t, nw_se = newey_west_t(daily_pnls, nlags=5)
        print(f"Test mean IC {mean_ic:.5f} | Test daily mean excess {daily_pnls.mean():.5e} | NW t {nw_t:.3f}")
        results.append({
            "train_start": train_dates[0], "train_end": train_dates[-1],
            "test_start": test_dates[0], "test_end": test_dates[-1],
            "mean_ic": mean_ic,
            "daily_excess_mean": float(daily_pnls.mean()),
            "nw_t": float(nw_t)
        })
        # Optionally break after one split for fast demo
        break
    return results

# --------------------------
# Main entry
# --------------------------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Please provide data at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    global FEATURE_COLS
    if FEATURE_COLS is None:
        # infer: all columns between STOCK_COL and FUTURE_RET_COL not including them
        cols = list(df.columns)
        si = cols.index(STOCK_COL)
        fi = cols.index(FUTURE_RET_COL)
        FEATURE_COLS = cols[si + 1 : fi]
        print("Inferred feature columns:", FEATURE_COLS)
    # basic scaling per feature cross-sectionally? Here we standardize per stock across time optionally
    for c in FEATURE_COLS:
        df[c] = df.groupby(STOCK_COL)[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    results = train_and_evaluate(df, FEATURE_COLS)
    print("Walk-forward results:", results)

if __name__ == "__main__":
    main()
